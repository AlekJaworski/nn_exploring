#!/usr/bin/env Rscript
# extract_phi_joint_parity.R
#
# Gaussian dispersion (φ = σ²) joint-outer-Newton parity capture.
#
# Demonstrates the gap between mgcv's REML (φ profiled out — search vector is
# just log λ) and ML (φ jointly optimised — search vector is [log λ, log φ]).
# This is the cleanest baseline for the planned mgcv_rust joint outer Newton
# generalisation (currently TDist-only at src/gam_optimized.rs:270-720 / the
# scat profile-shape block in src/smooth.rs).
#
# Outputs:
#   test_data/gaussian_phi_joint_parity.json   (Rust-readable parity payload)
#   test_data/gaussian_phi_joint_parity.rds    (full mgcv fit objects, optional)
#
# Reproducibility: set.seed(42); n=500; y = sin(2 pi x) + N(0, 0.4²).
# See docs/JOINT_OUTER_NEWTON_DESIGN.md for the planned consumer.

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

set.seed(42)
n  <- 500
x  <- runif(n)
mu_true <- sin(2 * pi * x)
sd_true <- 0.4
y  <- mu_true + rnorm(n, sd = sd_true)

# Common spec: cubic regression spline, 10 knots, gaussian identity. The
# difference is solely the smoothing-parameter selection method.
# control: dump diagnostic info that mgcv tucks away into $outer.info and
# $gcv.ubre.dev (the per-iter score trajectory).
ctrl <- gam.control(trace = FALSE, mgcv.tol = 1e-7, mgcv.half = 15,
                    keepData = TRUE)

cat("Fitting REML (phi profiled out of the search vector)...\n")
t_reml <- system.time({
  m_reml <- gam(y ~ s(x, bs = "cr", k = 10), method = "REML", control = ctrl)
})
cat(sprintf("  done in %.3fs\n", t_reml["elapsed"]))

cat("Fitting ML (phi joint with lambda in the search vector)...\n")
t_ml <- system.time({
  m_ml <- gam(y ~ s(x, bs = "cr", k = 10), method = "ML", control = ctrl)
})
cat(sprintf("  done in %.3fs\n", t_ml["elapsed"]))

# -- score trajectory (per-iter REML/ML criterion values) -----------------
# mgcv stashes the per-outer-iter score history in $outer.info$conv (only
# present when newton actually iterated). The score itself can also be
# re-evaluated at the final lambda via $gcv.ubre. We capture both.
extract_outer <- function(mod) {
  oi <- mod$outer.info
  list(
    iter            = if (!is.null(oi$iter)) oi$iter else NA_integer_,
    score           = mod$gcv.ubre,                # converged score
    score_method    = if (!is.null(oi$method)) oi$method else mod$method,
    conv            = if (!is.null(oi$conv)) oi$conv else NA_character_,
    # score evaluation history (last entries before stop). mgcv stores
    # the search-vector trace in $outer.info$sp (per-iter sp values) and
    # the per-iter score in $outer.info$score (also "$obj").
    sp_trajectory   = if (!is.null(oi$sp)) {
      if (is.list(oi$sp)) lapply(oi$sp, as.numeric) else as.numeric(oi$sp)
    } else NULL,
    score_trajectory= if (!is.null(oi$score)) as.numeric(oi$score) else NULL,
    obj_trajectory  = if (!is.null(oi$obj))   as.numeric(oi$obj)   else NULL,
    grad_at_conv    = if (!is.null(oi$grad)) as.numeric(oi$grad)   else NULL,
    hess_at_conv    = if (!is.null(oi$hess)) as.matrix(oi$hess)    else NULL
  )
}

# -- Joint REML1/REML2 probe at converged point via score-evaluator FD ----
#
# mgcv's gam.fit3 source shows: when scale is unknown AND scoreType in
# c("REML","ML","EFS"), the LAST element of the sp argument is log(scale).
# I.e. the joint outer-Newton search vector IS [log lambda, log phi].
# `$outer.info$grad` / `$hess` is the *stripped* M-dim view — mgcv reports
# at the M-dim manifold after profiling phi back out at convergence.
#
# To capture the true (M+1)-dim joint gradient/Hessian for the ML case, we
# FD on a score evaluator that uses fixed sp / scale. mgcv exposes this via
# `gam(..., sp = ..., scale = ..., method = "ML")`. With sp fixed and scale
# > 0, mgcv evaluates the inner PIRLS and returns $gcv.ubre (the score). We
# perturb the joint vector around the converged point and form central
# differences. (mgcv's own newton() uses analytical derivatives; this FD
# probe is purely for offline parity verification — slow but exact.)
probe_gradients_joint <- function(mod, method, log_sp_at, log_phi_at) {
  fdat <- mod$model
  # mgcv reports $sig2 as the post-PIRLS Pearson scale estimate; under ML
  # the joint-Newton optimum's log(phi) at the reported sp may differ by a
  # few percent. For an honest FD probe we *re-minimise* phi at the reported
  # sp first (1-D bracketed search), so theta0 lands on a stationary point
  # of the score surface — at which point the FD gradient should be ≈ 0.
  if (method == "ML") {
    sp_v <- as.numeric(mod$sp)
    eval_phi <- function(lp) {
      m <- tryCatch(
        gam(mod$formula, data = fdat, family = mod$family,
            method = method, sp = sp_v, scale = exp(lp), fit = TRUE),
        error = function(e) NULL
      )
      if (is.null(m)) Inf else as.numeric(m$gcv.ubre)
    }
    opt <- tryCatch(
      optimize(eval_phi, c(log_phi_at - 0.5, log_phi_at + 0.5), tol = 1e-9),
      error = function(e) NULL
    )
    log_phi_used <- if (!is.null(opt)) opt$minimum else log_phi_at
  } else {
    log_phi_used <- log_phi_at
  }

  theta0 <- if (method == "REML") log_sp_at else c(log_sp_at, log_phi_used)
  k <- length(theta0)
  h <- 1e-4

  # Score evaluator: refit with sp / scale fixed and read off the score.
  # mgcv allows `sp` to fix lambdas and `scale` to fix phi (scale > 0).
  # method controls which formula is used (REML vs ML) — both share the
  # same family + design.
  eval_score <- function(theta) {
    sp_v <- exp(theta[seq_len(length(mod$sp))])
    scale_v <- if (method == "REML") mod$sig2 else exp(theta[length(theta)])
    m <- tryCatch(
      gam(mod$formula, data = fdat, family = mod$family,
          method = method, sp = sp_v, scale = scale_v,
          fit = TRUE),
      error = function(e) NULL
    )
    if (is.null(m)) return(NA_real_)
    as.numeric(m$gcv.ubre)
  }

  f0 <- eval_score(theta0)
  grad <- rep(NA_real_, k)
  hess <- matrix(NA_real_, k, k)

  for (i in seq_len(k)) {
    tp <- theta0; tp[i] <- tp[i] + h
    tm <- theta0; tm[i] <- tm[i] - h
    fp <- eval_score(tp); fm <- eval_score(tm)
    grad[i] <- (fp - fm) / (2 * h)
    hess[i, i] <- (fp - 2 * f0 + fm) / (h * h)
  }
  for (i in seq_len(k)) for (j in seq_len(i - 1L)) {
    tpp <- theta0; tpp[i] <- tpp[i] + h; tpp[j] <- tpp[j] + h
    tpm <- theta0; tpm[i] <- tpm[i] + h; tpm[j] <- tpm[j] - h
    tmp <- theta0; tmp[i] <- tmp[i] - h; tmp[j] <- tmp[j] + h
    tmm <- theta0; tmm[i] <- tmm[i] - h; tmm[j] <- tmm[j] - h
    fpp <- eval_score(tpp); fpm <- eval_score(tpm)
    fmp <- eval_score(tmp); fmm <- eval_score(tmm)
    hess[i, j] <- hess[j, i] <- (fpp - fpm - fmp + fmm) / (4 * h * h)
  }

  list(
    method   = method,
    theta0   = theta0,
    score0   = f0,
    grad_FD  = grad,
    hess_FD  = hess,
    n_dim    = k,
    fd_h     = h,
    note = if (method == "ML") {
      "joint (M+1)-dim gradient/Hessian: last row+col are d/d log phi"
    } else {
      "M-dim only — phi profiled out of REML score"
    }
  )
}

# Build the JSON payload.
pack <- function(mod) {
  bcoef <- as.numeric(coef(mod))
  vp_diag <- as.numeric(diag(mod$Vp))
  ed <- as.numeric(mod$edf)
  list(
    method        = mod$method,
    sp            = as.numeric(mod$sp),       # smoothing parameters (length M)
    log_sp        = as.numeric(log(mod$sp)),
    sig2          = as.numeric(mod$sig2),     # dispersion estimate phi
    scale         = as.numeric(mod$scale),    # alias used in some places
    edf           = ed,
    edf_sum       = sum(ed),
    coef          = bcoef,
    coef_names    = names(coef(mod)),
    se            = sqrt(pmax(vp_diag, 0)),
    deviance      = as.numeric(mod$deviance),
    null_deviance = as.numeric(mod$null.deviance),
    aic           = as.numeric(mod$aic),
    gcv_ubre      = as.numeric(mod$gcv.ubre),
    rank          = as.integer(mod$rank),
    n             = as.integer(nrow(mod$model)),
    n_smooth_terms = length(mod$smooth),
    outer_info    = extract_outer(mod),
    derivs        = probe_gradients_joint(mod, mod$method,
                                          log(as.numeric(mod$sp)),
                                          log(as.numeric(mod$sig2)))
  )
}

payload <- list(
  meta = list(
    script        = "scripts/r/tests/extract_phi_joint_parity.R",
    purpose       = "Gaussian phi joint-outer-Newton (REML vs ML) parity baseline",
    mgcv_version  = as.character(packageVersion("mgcv")),
    r_version     = R.version.string,
    seed          = 42L,
    n             = n,
    sd_true       = sd_true,
    formula       = "y ~ s(x, bs='cr', k=10)",
    timing_reml_s = unname(t_reml["elapsed"]),
    timing_ml_s   = unname(t_ml["elapsed"])
  ),
  reml = pack(m_reml),
  ml   = pack(m_ml),
  # The headline diff: ML's outer search has one more dimension (log phi).
  search_vector_layout = list(
    reml = list(
      length = length(m_reml$sp),
      entries = "[log lambda_1, ..., log lambda_M]",
      phi_treatment = "profiled out of score; closed-form phi_hat = RSS / (n - tr(A))"
    ),
    ml = list(
      length = length(m_ml$sp) + 1L,
      entries = "[log lambda_1, ..., log lambda_M, log phi]",
      phi_treatment = "jointly optimised by outer Newton; gradient row n_minus_r/2 - ..."
    )
  )
)

out_json <- "test_data/gaussian_phi_joint_parity.json"
out_rds  <- "test_data/gaussian_phi_joint_parity.rds"

dir.create("test_data", showWarnings = FALSE, recursive = TRUE)
write_json(payload, out_json, pretty = TRUE, auto_unbox = TRUE, digits = 12,
           null = "null", na = "null")
saveRDS(list(reml = m_reml, ml = m_ml, payload = payload), out_rds)

cat(sprintf("\nWrote %s\n", out_json))
cat(sprintf("Wrote %s (full fit objects)\n", out_rds))

# Sanity print: side-by-side comparison.
cat("\n=== REML vs ML at convergence ===\n")
cat(sprintf("  sp (REML) : %s\n", paste(sprintf("%.6e", m_reml$sp), collapse = ", ")))
cat(sprintf("  sp (ML)   : %s\n", paste(sprintf("%.6e", m_ml$sp),   collapse = ", ")))
cat(sprintf("  sig2 (REML): %.6e\n", m_reml$sig2))
cat(sprintf("  sig2 (ML)  : %.6e\n", m_ml$sig2))
cat(sprintf("  edf sum (REML): %.4f\n", sum(m_reml$edf)))
cat(sprintf("  edf sum (ML)  : %.4f\n", sum(m_ml$edf)))
cat(sprintf("  score (REML): %.6f\n", m_reml$gcv.ubre))
cat(sprintf("  score (ML)  : %.6f\n", m_ml$gcv.ubre))
if (!is.null(m_reml$outer.info$iter)) {
  cat(sprintf("  outer iters (REML): %d\n", m_reml$outer.info$iter))
}
if (!is.null(m_ml$outer.info$iter)) {
  cat(sprintf("  outer iters (ML)  : %d\n", m_ml$outer.info$iter))
}
