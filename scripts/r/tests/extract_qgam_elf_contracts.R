#!/usr/bin/env Rscript
# Extract tiny, deterministic qgam::elf scalar-family contracts.
#
# This is intentionally below fit-level parity: it captures the per-row numbers
# that mgcv's gam.fit5 consumes on every iteration. Rust should match these
# before we touch warm starts, smoothing-parameter tuning, or production fits.

suppressPackageStartupMessages({
  library(qgam)
  library(mgcv)
  library(jsonlite)
})

repo_root <- normalizePath(getwd(), mustWork = TRUE)
out_path <- file.path(repo_root, "test_data", "qgam_elf_contracts.json")

tau <- 0.95
co <- 0.0031251342630865355
theta <- -4.9598429613200636
sigma <- exp(theta)

rows <- data.frame(
  name = c("near_zero_neg", "near_zero_pos", "left_tail", "right_tail", "centerish"),
  y = c(0.010, 0.030, -0.040, 0.110, 0.045),
  mu = c(0.020, 0.020, 0.015, 0.040, 0.043),
  wt = rep(1.0, 5)
)

fam <- qgam::elf(qu = tau, co = co, theta = theta)

row_payload <- lapply(seq_len(nrow(rows)), function(i) {
  y <- rows$y[i]
  mu <- rows$mu[i]
  wt <- rows$wt[i]
  dd0 <- fam$Dd(y, mu, theta, wt, level = 0)
  dd1 <- fam$Dd(y, mu, theta, wt, level = 1)
  dd2 <- fam$Dd(y, mu, theta, wt, level = 2)
  list(
    name = rows$name[i],
    y = y,
    mu = mu,
    wt = wt,
    dev_resids = as.numeric(fam$dev.resids(y, mu, wt, theta)),
    Dmu = as.numeric(dd0$Dmu),
    Dmu2 = as.numeric(dd0$Dmu2),
    EDmu2 = as.numeric(dd0$EDmu2),
    Dth = as.numeric(dd1$Dth),
    Dmuth = as.numeric(dd1$Dmuth),
    Dmu3 = as.numeric(dd1$Dmu3),
    Dmu2th = as.numeric(dd1$Dmu2th),
    Dmu4 = as.numeric(dd2$Dmu4),
    Dth2 = as.numeric(dd2$Dth2),
    Dmuth2 = as.numeric(dd2$Dmuth2),
    Dmu2th2 = as.numeric(dd2$Dmu2th2),
    Dmu3th = as.numeric(dd2$Dmu3th)
  )
})

ls_payload <- fam$ls(rows$y, rows$wt, theta, scale = 1)

payload <- list(
  source = "qgam::elf scalar contract",
  versions = list(
    qgam = as.character(packageVersion("qgam")),
    mgcv = as.character(packageVersion("mgcv")),
    R = as.character(getRversion())
  ),
  params = list(
    tau = tau,
    co = co,
    theta = theta,
    sigma = sigma
  ),
  rows = row_payload,
  ls = list(
    value = as.numeric(ls_payload$ls),
    lsth1 = as.numeric(ls_payload$lsth1),
    lsth2 = as.numeric(ls_payload$lsth2)
  )
)

# Diagnostic-only nat.param / diagonal.penalty snapshot for the production
# fixed-sp fixture. This is intentionally not consumed by production code; it
# freezes qgam's penalty-coordinate objects before the ELF Newton work starts.
sale_dir <- file.path(repo_root, "data", "sale_price_fixtures")
train_path <- file.path(sale_dir, "entire_dataset_train.parquet")
mean_path <- file.path(sale_dir, "mgcv_rust_parity", "mean_gam_entire_dataset.parquet")
if (requireNamespace("arrow", quietly = TRUE) && file.exists(train_path) && file.exists(mean_path)) {
  train_df <- as.data.frame(arrow::read_parquet(train_path))
  mean_ref <- as.data.frame(arrow::read_parquet(mean_path))
  train_df$resid <- mean_ref$y - mean_ref$pred_prod
  k_map <- list(
    days_in_current_price_regime = 5L,
    cum_dom_before_current_regime = 5L,
    price_change_pct_from_original = 5L,
    monthly_index = 5L
  )
  form <- resid ~ s(days_in_current_price_regime, k = 5, bs = "cr") +
    s(cum_dom_before_current_regime, k = 5, bs = "cr") +
    s(price_change_pct_from_original, k = 5, bs = "cr") +
    s(monthly_index, k = 5, bs = "cr")
  qfit <- qgam::qgam(form, data = train_df, qu = tau, lsig = theta, err = co)
  payload$production_natparam <- list(
    source = "qgam production fixed-sp nat.param / diagonal.penalty diagnostic",
    n = nrow(train_df),
    k = k_map,
    sp = as.numeric(qfit$sp),
    min_sp = as.numeric(qfit$min.sp),
    coefficients = as.numeric(coef(qfit)),
    diagonal_penalty = if (!is.null(qfit$diagonal.penalty)) as.numeric(qfit$diagonal.penalty) else NULL,
    nat_param = if (!is.null(qfit$nat.param)) as.numeric(qfit$nat.param) else NULL
  )
}

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(payload, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 17)
cat(sprintf("Wrote %s\n", out_path))
