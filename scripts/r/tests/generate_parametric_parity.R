# scripts/r/tests/generate_parametric_parity.R
#
# Generate mgcv-side parity reference for `y ~ s(x) + dummy` with a binary
# `dummy` covariate. Captures everything needed to verify both final
# coefficients AND structural correctness (design layout, per-term EDF,
# lambda, sigma^2) — see docs/PARAMETRIC_TERMS_DESIGN.md for the protocol.
#
# Output: test_data/parametric_parity_n300.json
#
# To regenerate:  Rscript scripts/r/tests/generate_parametric_parity.R

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

set.seed(42)
n <- 300L

x <- runif(n)
dummy <- as.integer(runif(n) < 0.5)
# True DGP: smooth in x + linear shift on dummy + small noise.
# dummy_effect = 0.7 — large enough to be detected against 0.1 noise.
y <- sin(2 * pi * x) + 0.7 * dummy + rnorm(n, sd = 0.1)

# Fit gam(y ~ s(x) + dummy, method="REML"). We use k=10 (mgcv's default for
# univariate smooths) so the parity test against our Rust path can use
# k=[10, 1] (10 for s(x), 1 placeholder for the parametric column).
m <- gam(y ~ s(x, k = 10) + dummy, method = "REML")

# ---- extract everything --------------------------------------------------- #

coef_named <- as.list(coef(m))            # named list: "(Intercept)", "dummy", "s(x).1", ...
sp_named   <- as.list(m$sp)               # named list: one entry per smooth — "s(x)"
edf_named  <- as.list(m$edf)              # named vector, one entry per coef column
# Per-term EDF (compact): mgcv's m$edf is per-column; m$edf1 is per-term
# aggregated. summary.gam(m)$s.table has the smooth-only summary; we expose
# both for clarity.
sum_m <- summary(m)
edf_smooth <- as.list(sum_m$s.table[, "edf"])   # named: "s(x)"

# Design matrix (n x p) — mgcv's lpmatrix. This is the canonical layout
# we want to match on the Rust side.
lp <- predict(m, type = "lpmatrix")
lp_colnames <- colnames(lp)
lp_dims <- dim(lp)

# Fitted values on the response scale (Gaussian => same as link scale).
fitted_vals <- as.numeric(fitted(m))

# Sigma^2 (scale parameter).
sigma2 <- as.numeric(m$sig2)

# Edf totals — useful for the sanity check "parametric edf = 1.0".
edf_total <- sum(unlist(m$edf))

# ---- assemble JSON payload ------------------------------------------------ #

payload <- list(
  meta = list(
    formula  = "y ~ s(x, k = 10) + dummy",
    method   = "REML",
    n        = n,
    k_smooth = 10L,
    seed     = 42L,
    mgcv_version = as.character(packageVersion("mgcv"))
  ),
  # raw training data — Rust side fits the same model
  data = list(
    x     = x,
    dummy = dummy,
    y     = y
  ),
  # mgcv outputs
  mgcv = list(
    coef       = coef_named,            # named: intercept, dummy, s(x).1..9
    sp         = sp_named,              # named: s(x) -> lambda
    edf_by_col = edf_named,             # per-column EDF (length 11 for k=10)
    edf_smooth = edf_smooth,            # per-smooth summary EDF
    edf_total  = edf_total,
    sigma2     = sigma2,
    fitted     = fitted_vals,
    lpmatrix_colnames = lp_colnames,    # column ordering — verify structural parity
    lpmatrix_dims     = lp_dims         # (n, p)
  )
)

out_path <- "test_data/parametric_parity_n300.json"
write(toJSON(payload, pretty = TRUE, digits = 17, auto_unbox = TRUE), out_path)

cat("--- mgcv parametric parity reference -----------------------------------\n")
cat(sprintf("n            = %d\n", n))
cat(sprintf("mgcv version = %s\n", as.character(packageVersion("mgcv"))))
cat(sprintf("intercept    = %.10g\n", coef(m)["(Intercept)"]))
cat(sprintf("dummy coef   = %.10g  (true = 0.7)\n", coef(m)["dummy"]))
cat(sprintf("lambda s(x)  = %.10g\n", m$sp["s(x)"]))
cat(sprintf("sigma^2      = %.10g  (true noise sd^2 = 0.01)\n", sigma2))
cat(sprintf("EDF s(x)     = %.4f\n", sum_m$s.table["s(x)", "edf"]))
cat(sprintf("EDF dummy    = %.4f  (should be ~1.0)\n", m$edf[2]))
cat(sprintf("design dims  = %d x %d\n", lp_dims[1], lp_dims[2]))
cat(sprintf("design cols  = %s\n", paste(lp_colnames, collapse = ", ")))
cat(sprintf("\nWrote %s\n", out_path))
