#!/usr/bin/env Rscript
# Capture ocat parity data — final fit + per-iteration trajectory.
#
# Captures:
#   * coefficients (β) — smooth-coefs only; thresholds θ live in family$getTheta()
#   * thresholds  (θ, raw and cumulatively transformed)
#   * fitted per-category probabilities (n × R)
#   * fitted linear predictor η
#   * working residuals
#   * design matrix layout (column names, dims)
#   * sp (λ) — should be length 1 (1 smooth, no select)
#   * edf per smooth (m$edf, m$edf1, m$edf2)
#   * deviance, REML score
#   * λ trajectory across outer iterations (trace=TRUE captures sp at each)
#
# Output: /home/alex/vibe_coding/nn_exploring/test_data/ocat_parity_basic.json

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

set.seed(42)
n <- 500
x <- runif(n)
eta_true <- 2 * sin(2 * pi * x)

# 4 categories from cumulative thresholds — matches the brief.
# probs[, k] = P(y == k).  Note mgcv's ocat indexes categories 1..R.
R_levels <- 4
thresh_true <- c(-1.5, 0, 1.5)
probs <- t(sapply(eta_true, function(e) {
  p_le <- plogis(thresh_true - e)            # P(Y <= k), k=1..R-1
  c(p_le[1], diff(c(p_le, 1)))               # category probs
}))
y <- apply(probs, 1, function(p) sample(seq_len(R_levels), 1, prob = p))

df <- data.frame(x = x, y = y)

# Capture λ-trajectory via gam.control(trace=TRUE). We post-process the trace
# output but mgcv stores intermediate sp in fit$outer.info$sp.full (when
# method="REML") — capture both.
ctrl <- gam.control(trace = FALSE, epsilon = 1e-9, maxit = 200)
m <- gam(y ~ s(x, k = 10, bs = "cr"),
         data    = df,
         family  = ocat(R = R_levels),
         method  = "REML",
         control = ctrl)

# Some extended families store iteration trajectory via outer.info; for
# REML/Newton it's in $outer.info$sp.full and $outer.info$score.hist when
# available. mgcv 1.9 exposes $outer.info$conv but not always sp.full —
# fall back to printing what is there.
outer_info <- m$outer.info

# Fitted per-category probabilities — re-derive using mgcv's threshold layout.
# Cumulative thresholds in mgcv's internal layout:
#   alpha[1] = -Inf, alpha[2] = -1, alpha[k] = alpha[2] + cumsum(exp(theta))[k-2]
#   alpha[R+1] = Inf.  P(Y == k) = F(alpha[k+1] - η) - F(alpha[k] - η).
theta_raw <- m$family$getTheta(trans = FALSE)         # log-gaps (length R-2)
theta_alpha <- m$family$getTheta(trans = TRUE)        # cumulative thresholds (length R-1)
eta_hat <- as.numeric(m$linear.predictors)
mu_hat  <- as.numeric(m$fitted.values)                # for identity link, == eta

# Build full alpha series:
alpha_full <- c(-Inf, theta_alpha, Inf)               # length R+1

prob_mat <- matrix(0, n, R_levels)
for (k in seq_len(R_levels)) {
  prob_mat[, k] <- plogis(alpha_full[k + 1] - eta_hat) -
                   plogis(alpha_full[k]     - eta_hat)
}

# Design matrix + penalty + structural diagnostics
X_design <- model.matrix(m)
# m$smooth[[1]]$S is the penalty matrix block for s(x).
S1 <- m$smooth[[1]]$S[[1]]

out <- list(
  meta = list(
    family   = "ocat",
    R_levels = R_levels,
    n        = n,
    seed     = 42,
    method   = "REML",
    mgcv_version = as.character(packageVersion("mgcv"))
  ),
  # Data
  x = x,
  y = y,
  # True data-generating params (ground truth for the test)
  truth = list(
    thresh = thresh_true,
    eta    = eta_true
  ),
  # Final fit — full state
  fit = list(
    coefficients     = as.numeric(coef(m)),
    coef_names       = names(coef(m)),
    theta_raw        = as.numeric(theta_raw),   # log-gaps (free params)
    theta_alpha      = as.numeric(theta_alpha), # cumulative thresholds (R-1)
    eta              = eta_hat,
    fitted_prob      = prob_mat,                # n × R
    working_residual = as.numeric(residuals(m, type = "working")),
    deviance_residual= as.numeric(residuals(m, type = "deviance")),
    response_residual= as.numeric(residuals(m, type = "response")),
    sp               = as.numeric(m$sp),        # smoothing params (linear scale)
    log_sp           = as.numeric(log(m$sp)),
    edf              = as.numeric(m$edf),       # per-coef edf
    edf1             = as.numeric(m$edf1),      # alt edf
    edf2             = as.numeric(m$edf2),      # alt edf
    edf_per_smooth   = as.numeric(summary(m)$s.table[, "edf"]),
    deviance         = as.numeric(m$deviance),
    null_deviance    = as.numeric(m$null.deviance),
    aic              = as.numeric(m$aic),
    reml_score       = as.numeric(m$gcv.ubre),
    iterations       = if (!is.null(outer_info$iter)) outer_info$iter else NA,
    converged        = m$converged
  ),
  # Outer-loop trajectory (may be NA depending on family/method path)
  trajectory = list(
    sp_hist    = if (!is.null(outer_info$sp.full)) outer_info$sp.full else NA,
    score_hist = if (!is.null(outer_info$score.hist)) outer_info$score.hist else NA,
    grad       = if (!is.null(outer_info$grad)) outer_info$grad else NA,
    hess       = if (!is.null(outer_info$hess)) outer_info$hess else NA
  ),
  # Structural diagnostics
  structure = list(
    design_dim     = dim(X_design),
    design_colnames= colnames(X_design),
    design_X       = X_design,         # save full design matrix for parity
    penalty_S      = S1,
    penalty_rank   = as.integer(m$smooth[[1]]$rank),
    null_space_dim = as.integer(m$smooth[[1]]$null.space.dim),
    p              = ncol(X_design),
    Mp             = as.integer(sum(m$paraPen$Mp))  # may be NULL
  ),
  family_info = list(
    n_theta        = m$family$n.theta,
    ini_theta      = m$family$ini.theta
  )
)

out_path <- "/home/alex/vibe_coding/nn_exploring/test_data/ocat_parity_basic.json"
write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE,
           digits = NA, matrix = "rowmajor", na = "null")
cat(sprintf("Wrote %s\n", out_path))

# Brief console summary so the captured state is obvious.
cat("\n== ocat parity summary ==\n")
cat(sprintf("R = %d categories, n = %d\n", R_levels, n))
cat(sprintf("sp = %s\n", paste(sprintf("%.6g", m$sp), collapse = ", ")))
cat(sprintf("theta (free, log-gaps) = %s\n", paste(sprintf("%.6g", theta_raw), collapse = ", ")))
cat(sprintf("theta (cumulative alpha) = %s  (truth: %s)\n",
            paste(sprintf("%.4f", theta_alpha), collapse = ", "),
            paste(sprintf("%.4f", thresh_true), collapse = ", ")))
cat(sprintf("edf (sum) = %.4f, deviance = %.4f, REML = %.4f\n",
            sum(m$edf), m$deviance, m$gcv.ubre))
