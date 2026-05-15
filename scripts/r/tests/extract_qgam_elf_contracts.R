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

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(payload, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 17)
cat(sprintf("Wrote %s\n", out_path))
