#!/usr/bin/env Rscript
# Extract qgam .getErrParam intermediates for the production entire_dataset fixture.
#
# Captures: standardized-residual summary, n, d (effective df), SHASH params,
# pmode, per-quantile (qhat, lf0, lf1, h, err) so Python can verify the
# SHASH math and err/co computation before changing any fit-level behaviour.

suppressPackageStartupMessages({
  library(qgam)
  library(mgcv)
  library(nanoparquet)
  library(jsonlite)
})

repo_root <- normalizePath(getwd(), mustWork = TRUE)
fixture_dir <- file.path(repo_root, "data", "sale_price_fixtures")
parity_dir  <- file.path(fixture_dir, "mgcv_rust_parity")
out_path    <- file.path(repo_root, "test_data", "qgam_err_co_contract.json")

target <- "sale_to_list_price_ratio"
smooths <- c(
  "days_in_current_price_regime",
  "cum_dom_before_current_regime",
  "price_change_pct_from_original",
  "monthly_index"
)
tau <- 0.95

# ── load data ────────────────────────────────────────────────────────────────
df     <- read_parquet(file.path(fixture_dir, "entire_dataset_train.parquet"))
ref    <- read_parquet(file.path(parity_dir, "mean_gam_entire_dataset.parquet"))
df$resid <- df[[target]] - ref$pred_prod

k_map <- setNames(rep(5L, length(smooths)), smooths)
for (col in smooths) {
  k_map[col] <- max(3L, min(5L, length(unique(df[[col]]))))
}

formula_str <- paste0("resid ~ ",
  paste(sprintf("s(%s, k=%d, bs='cr')", smooths, k_map[smooths]), collapse=" + "))
formula_obj <- as.formula(formula_str)

cat("Formula:", formula_str, "\n")
cat("n =", nrow(df), "\n")

# ── fit Gaussian GAM (same as .init_gauss_fit) ───────────────────────────────
gFit <- gam(formula_obj, data = df, method = "REML")
varHat <- gFit$sig2
cat("varHat =", varHat, "\n")

# ── .getErrParam internals ───────────────────────────────────────────────────
muHat <- as.numeric(gFit$fitted.values)
r <- (df$resid - muHat) / sqrt(varHat)
n <- length(r)

anv <- anova(gFit)
d_param <- sum(anv$pTerms.df[!grepl("\\.1", rownames(anv$pTerms.table))]) +
           as.integer("(Intercept)" %in% names(anv$p.coeff))
d_smooth <- sum(unique(pen.edf(gFit)[!grepl("s\\.1|te\\.1|ti\\.1|t2\\.1",
                                             names(pen.edf(gFit)))]))
d <- d_param + d_smooth
cat("d_param =", d_param, "d_smooth =", d_smooth, "d =", d, "\n")

fitSH <- qgam:::.fitShash(r)
parSH <- fitSH$par
cat("parSH =", parSH, "\n")
cat("fitShash convergence =", fitSH$convergence, "\n")

pmode_x <- qgam:::.shashMode(parSH)
pmode   <- qgam:::.shashCDF(pmode_x, parSH)
cat("pmode =", pmode, "\n")

# per-quantile payload
qu_list <- c(tau)
qu_payload <- lapply(qu_list, function(qu) {
  quX <- qu
  if (abs(quX - pmode) < 0.05) {
    quX <- pmode + sign(quX - pmode) * 0.05
    quX <- max(min(quX, 0.99), 0.01)
  }
  qhat <- qgam:::.shashQf(quX, parSH)
  lf0  <- qgam:::.llkShash(qhat, mu=parSH[1], tau=parSH[2],
                            eps=parSH[3], phi=parSH[4])$l0
  lf1_raw <- -qgam:::.llkShash(qhat, mu=parSH[1], tau=parSH[2],
                                eps=parSH[3], phi=parSH[4], deriv=1)$l1[1]
  lf1 <- log(abs(lf1_raw)) + lf0
  h   <- (d * 9 / (n * pi^4))^(1/3) * exp(lf0/3 - 2*lf1/3)
  err <- min(h * 2 * log(2) / sqrt(2 * pi), 1.0)
  list(
    qu     = qu,
    quX    = quX,
    qhat   = qhat,
    lf0    = lf0,
    lf1_raw = lf1_raw,
    lf1    = lf1,
    h      = h,
    err    = err
  )
})

# call the actual .getErrParam to double-check
err_official <- qgam:::.getErrParam(qu = qu_list, gFit = gFit, varHat = varHat)
cat("err_official =", err_official, "\n")
cat("err_manual   =", sapply(qu_payload, `[[`, "err"), "\n")

# ── summary stats for standardized residuals (portable, no large array) ──────
r_summary <- list(
  min   = min(r),
  q25   = as.numeric(quantile(r, 0.25)),
  med   = median(r),
  q75   = as.numeric(quantile(r, 0.75)),
  max   = max(r),
  mean  = mean(r),
  sd    = sd(r),
  n     = n
)

payload <- list(
  source = "qgam:::.getErrParam production entire_dataset",
  versions = list(
    qgam = as.character(packageVersion("qgam")),
    mgcv = as.character(packageVersion("mgcv")),
    R    = as.character(getRversion())
  ),
  formula = formula_str,
  n = n,
  varHat = varHat,
  r_summary = r_summary,
  d_param  = d_param,
  d_smooth = d_smooth,
  d        = d,
  parSH    = as.numeric(parSH),
  fitShash_convergence = fitSH$convergence,
  pmode_x  = pmode_x,
  pmode    = pmode,
  qu_contracts = qu_payload,
  err_official = as.numeric(err_official)
)

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(payload, out_path, auto_unbox = TRUE, pretty = TRUE, digits = 17)
cat(sprintf("Wrote %s\n", out_path))
