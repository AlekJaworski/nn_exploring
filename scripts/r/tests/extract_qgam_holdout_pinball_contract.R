#!/usr/bin/env Rscript
# Extract out-of-sample qgam predictions for the holdout pinball contract.
#
# Fits qgam on the first 80% of entire_dataset_train rows, predicts on the
# last 20%, and saves the OOS predictions so test_quantile_eval.py can make
# a fair Rust vs qgam pinball comparison (both out-of-sample).

suppressPackageStartupMessages({
  library(qgam)
  library(mgcv)
  library(nanoparquet)
  library(jsonlite)
})

repo_root <- normalizePath(getwd(), mustWork = TRUE)
fixture_dir <- file.path(repo_root, "data", "sale_price_fixtures")
parity_dir  <- file.path(fixture_dir, "mgcv_rust_parity")
out_path    <- file.path(repo_root, "test_data", "qgam_holdout_pinball_contract.json")

target  <- "sale_to_list_price_ratio"
smooths <- c(
  "days_in_current_price_regime",
  "cum_dom_before_current_regime",
  "price_change_pct_from_original",
  "monthly_index"
)
tau  <- 0.95
k    <- 5L

# ── load data ────────────────────────────────────────────────────────────────
df  <- read_parquet(file.path(fixture_dir, "entire_dataset_train.parquet"))
ref <- read_parquet(file.path(parity_dir,  "mean_gam_entire_dataset.parquet"))
df$resid <- df[[target]] - ref$pred_prod

n       <- nrow(df)
split   <- floor(0.8 * n)
cat(sprintf("n=%d  split=%d  train=%d  test=%d\n", n, split, split, n - split))

df_train <- df[seq_len(split), ]
df_test  <- df[(split + 1):n, ]

# ── build formula ─────────────────────────────────────────────────────────────
k_map <- setNames(rep(k, length(smooths)), smooths)
for (col in smooths) {
  k_map[col] <- max(3L, min(k, length(unique(df_train[[col]]))))
}
formula_str <- paste0(
  "resid ~ ",
  paste(sprintf("s(%s, k=%d, bs='cr')", smooths, k_map[smooths]), collapse = " + ")
)
formula_obj <- as.formula(formula_str)
cat("Formula:", formula_str, "\n")

# ── fit qgam on training split ────────────────────────────────────────────────
cat("Fitting qgam on train split (n =", nrow(df_train), ")...\n")
t0 <- proc.time()["elapsed"]
fit <- qgam(formula_obj, data = df_train, qu = tau)
fit_s <- as.numeric(proc.time()["elapsed"] - t0)
cat(sprintf("qgam fit time: %.2fs\n", fit_s))

# ── predict on test split ────────────────────────────────────────────────────
pred_test <- as.numeric(predict(fit, newdata = df_test))
resid_test <- df_test$resid

# ── pinball loss ─────────────────────────────────────────────────────────────
pinball_loss <- function(y, pred, tau) {
  r <- y - pred
  mean(pmax(tau * r, (tau - 1) * r))
}
pb       <- pinball_loss(resid_test, pred_test, tau)
coverage <- mean(resid_test < pred_test)
cat(sprintf("qgam OOS pinball  (tau=%.2f): %.6f\n", tau, pb))
cat(sprintf("qgam OOS coverage (tau=%.2f): %.4f  (target %.4f)\n", tau, coverage, tau))

# ── save contract ─────────────────────────────────────────────────────────────
contract <- list(
  tau         = tau,
  n_total     = n,
  split_index = split,
  n_train     = split,
  n_test      = n - split,
  formula     = formula_str,
  fit_time_s  = fit_s,
  qgam_oos_pinball  = pb,
  qgam_oos_coverage = coverage,
  qgam_pred_test    = as.numeric(pred_test)
)

write(toJSON(contract, auto_unbox = TRUE, digits = 15), out_path)
cat("Written:", out_path, "\n")
