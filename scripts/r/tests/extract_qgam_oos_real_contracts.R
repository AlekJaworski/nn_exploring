#!/usr/bin/env Rscript
# Extract qgam out-of-sample predictions for multiple real sale-price fixtures.
#
# This extends the single holdout contract to a small real OOS battery:
# - entire_dataset ordered 80/20 split (same as qgam_holdout_pinball_contract)
# - entire_dataset seeded random 80/20 splits
# - split_0..split_4 ordered 80/20 internal holdouts

suppressPackageStartupMessages({
  library(qgam)
  library(mgcv)
  library(nanoparquet)
  library(jsonlite)
})

repo_root <- normalizePath(getwd(), mustWork = TRUE)
fixture_dir <- file.path(repo_root, "data", "sale_price_fixtures")
parity_dir  <- file.path(fixture_dir, "mgcv_rust_parity")
out_path <- Sys.getenv(
  "QGAM_OOS_OUT_PATH",
  unset = file.path(repo_root, "test_data", "qgam_oos_real_contracts.json")
)

target  <- "sale_to_list_price_ratio"
smooths <- c(
  "days_in_current_price_regime",
  "cum_dom_before_current_regime",
  "price_change_pct_from_original",
  "monthly_index"
)
tau <- 0.95
k <- 5L

parse_seed_spec <- function(spec) {
  if (grepl(":", spec, fixed = TRUE)) {
    parts <- strsplit(spec, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2L) stop("QGAM_RANDOM_SEEDS range must look like '0:19'")
    return(seq.int(as.integer(parts[[1]]), as.integer(parts[[2]])))
  }
  as.integer(strsplit(spec, ",", fixed = TRUE)[[1]])
}

random_seeds <- parse_seed_spec(Sys.getenv("QGAM_RANDOM_SEEDS", unset = "0,1,2"))

pinball_loss <- function(y, pred, tau) {
  r <- y - pred
  mean(pmax(tau * r, (tau - 1) * r))
}

build_formula <- function(df_train) {
  k_map <- setNames(rep(k, length(smooths)), smooths)
  for (col in smooths) {
    k_map[col] <- max(3L, min(k, length(unique(df_train[[col]]))))
  }
  paste0(
    "resid ~ ",
    paste(sprintf("s(%s, k=%d, bs='cr')", smooths, k_map[smooths]), collapse = " + ")
  )
}

load_case_df <- function(source_name, mean_name) {
  df <- read_parquet(file.path(fixture_dir, source_name))
  ref <- read_parquet(file.path(parity_dir, mean_name))
  df$resid <- df[[target]] - ref$pred_prod
  df
}

fit_contract <- function(case_id, df, train_idx, test_idx, split_kind, seed = NULL,
                         source_files = list()) {
  df_train <- df[train_idx, ]
  df_test <- df[test_idx, ]
  formula_str <- build_formula(df_train)
  formula_obj <- as.formula(formula_str)

  cat(sprintf("\n[%s] train=%d test=%d split=%s\n", case_id, nrow(df_train), nrow(df_test), split_kind))
  cat("Formula:", formula_str, "\n")
  t0 <- proc.time()["elapsed"]
  fit <- qgam(formula_obj, data = df_train, qu = tau)
  fit_s <- as.numeric(proc.time()["elapsed"] - t0)
  pred_test <- as.numeric(predict(fit, newdata = df_test))
  resid_test <- df_test$resid
  pb <- pinball_loss(resid_test, pred_test, tau)
  coverage <- mean(resid_test < pred_test)
  cat(sprintf("qgam fit=%.2fs pinball=%.8f coverage=%.4f\n", fit_s, pb, coverage))

  list(
    case_id = case_id,
    tau = tau,
    n_total = nrow(df),
    n_train = length(train_idx),
    n_test = length(test_idx),
    train_idx_1based = as.integer(train_idx),
    test_idx_1based = as.integer(test_idx),
    split_kind = split_kind,
    seed = seed,
    formula = formula_str,
    fit_time_s = fit_s,
    qgam_oos_pinball = pb,
    qgam_oos_coverage = coverage,
    qgam_pred_test = as.numeric(pred_test),
    source_files = source_files
  )
}

contracts <- list()

# Entire dataset, ordered 80/20 (matches the original single contract).
df_entire <- load_case_df("entire_dataset_train.parquet", "mean_gam_entire_dataset.parquet")
n <- nrow(df_entire)
split <- floor(0.8 * n)
contracts[[length(contracts) + 1L]] <- fit_contract(
  "sale_price_q95_contract_80_20",
  df_entire,
  seq_len(split),
  (split + 1L):n,
  "ordered_80_20",
  NULL,
  list("entire_dataset_train.parquet", "mgcv_rust_parity/mean_gam_entire_dataset.parquet")
)

# Entire dataset, random seeded 80/20 splits.
for (seed in random_seeds) {
  set.seed(seed)
  idx <- sample.int(n)
  train_idx <- sort(idx[seq_len(split)])
  test_idx <- sort(idx[(split + 1L):n])
  contracts[[length(contracts) + 1L]] <- fit_contract(
    sprintf("sale_price_entire_seed_%d_q95_random_80_20", seed),
    df_entire,
    train_idx,
    test_idx,
    "random_80_20",
    seed,
    list("entire_dataset_train.parquet", "mgcv_rust_parity/mean_gam_entire_dataset.parquet")
  )
}

# Production split fixtures, ordered internal 80/20 holdouts.
for (i in 0:4) {
  source_name <- sprintf("split_%d_train.parquet", i)
  mean_name <- sprintf("mean_gam_split_%d.parquet", i)
  df_split <- load_case_df(source_name, mean_name)
  n_i <- nrow(df_split)
  split_i <- floor(0.8 * n_i)
  contracts[[length(contracts) + 1L]] <- fit_contract(
    sprintf("sale_price_split_%d_q95_internal_80_20", i),
    df_split,
    seq_len(split_i),
    (split_i + 1L):n_i,
    "ordered_80_20",
    NULL,
    list(source_name, file.path("mgcv_rust_parity", mean_name))
  )
}

payload <- list(
  benchmark_version = "qgam_oos_real_contracts_v1",
  tau = tau,
  smooths = smooths,
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  contracts = contracts
)

write(toJSON(payload, auto_unbox = TRUE, digits = 15), out_path)
cat("\nWritten:", out_path, "\n")
