#!/usr/bin/env Rscript
# Capture select=TRUE parity data — null-space penalty doubling.
#
# Two fits side by side:
#   * m_no_select: standard GAM on (x1, x2). λ has length 2.
#   * m_select:    select=TRUE adds a null-space penalty per smooth. λ has
#                  length 4 — original + null-space penalty per smooth.
#
# x2 has NO effect on y; with select=TRUE its null-space λ should grow large,
# collapsing the smooth to zero.
#
# Captures:
#   * sp (λ) — length 2 vs length 4 (the structural difference)
#   * penalty matrices: m$smooth[[j]]$S = list of penalty blocks per smooth.
#     Under select=TRUE, length(m$smooth[[j]]$S) == 2 (S[[1]] = original,
#     S[[2]] = null-space penalty).
#   * coefficients, edf, fitted values
#
# Output:
#   /home/alex/vibe_coding/nn_exploring/test_data/select_parity_basic.json

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

set.seed(42)
n <- 400
x1 <- runif(n)
x2 <- runif(n)                            # NO effect on y
y  <- sin(2 * pi * x1) + rnorm(n, sd = 0.2)

df <- data.frame(x1 = x1, x2 = x2, y = y)

ctrl <- gam.control(trace = FALSE, epsilon = 1e-9, maxit = 200)

m_no_select <- gam(y ~ s(x1, k = 10, bs = "cr") + s(x2, k = 10, bs = "cr"),
                   data = df, method = "REML", control = ctrl)
m_select    <- gam(y ~ s(x1, k = 10, bs = "cr") + s(x2, k = 10, bs = "cr"),
                   data = df, method = "REML", select = TRUE, control = ctrl)

# Structural check first — this is the load-bearing assertion.
sp_no_select <- as.numeric(m_no_select$sp)
sp_select    <- as.numeric(m_select$sp)
stopifnot(length(sp_no_select) == 2)
stopifnot(length(sp_select)    == 4)        # 2 smooths × 2 penalties each

cat("\n== sp structural difference ==\n")
cat(sprintf("no-select sp (len %d): %s\n",
            length(sp_no_select),
            paste(sprintf("%.4g", sp_no_select), collapse = ", ")))
cat(sprintf("select    sp (len %d): %s\n",
            length(sp_select),
            paste(sprintf("%.4g", sp_select), collapse = ", ")))

# Per-smooth penalty structure under select=TRUE.
# m_select$smooth[[j]]$S is a list of length 2 — second block is the null-space
# projector U·U' (built by smoothCon at gam.fit5/gam.fit3.r:432-481).
sm_penalties <- function(m) {
  lapply(seq_along(m$smooth), function(j) {
    list(
      term      = m$smooth[[j]]$term,
      first.para= m$smooth[[j]]$first.para,
      last.para = m$smooth[[j]]$last.para,
      df        = m$smooth[[j]]$df,
      bs.dim    = m$smooth[[j]]$bs.dim,
      n_penalties = length(m$smooth[[j]]$S),
      ranks       = as.integer(m$smooth[[j]]$rank),
      null_space_dim = as.integer(m$smooth[[j]]$null.space.dim),
      S_list      = m$smooth[[j]]$S          # each is a matrix, jsonlite handles it
    )
  })
}

# Coef/edf split per smooth — under select=TRUE the smooth-2 (x2) coefs should
# shrink toward zero.
coef_per_smooth <- function(m) {
  out <- list()
  for (j in seq_along(m$smooth)) {
    rng <- m$smooth[[j]]$first.para:m$smooth[[j]]$last.para
    out[[j]] <- list(
      term     = m$smooth[[j]]$term,
      coef_idx = rng,
      coef     = as.numeric(coef(m)[rng]),
      coef_l2  = sqrt(sum(coef(m)[rng]^2))
    )
  }
  out
}

# Per-smooth fitted contribution — this is what we'd plot. Use predict(type="terms").
fitted_terms <- function(m, df) {
  pt <- predict(m, type = "terms", newdata = df)
  list(
    colnames = colnames(pt),
    values   = pt
  )
}

out <- list(
  meta = list(
    feature      = "select_true_null_space_penalty",
    n            = n,
    seed         = 42,
    method       = "REML",
    mgcv_version = as.character(packageVersion("mgcv"))
  ),
  x1 = x1, x2 = x2, y = y,
  truth = list(
    note = "y = sin(2*pi*x1) + noise; x2 has no effect"
  ),
  # The "no-select" reference fit (existing parity-quality baseline)
  no_select = list(
    sp                = sp_no_select,
    log_sp            = log(sp_no_select),
    coefficients      = as.numeric(coef(m_no_select)),
    coef_names        = names(coef(m_no_select)),
    edf               = as.numeric(m_no_select$edf),
    edf_per_smooth    = as.numeric(summary(m_no_select)$s.table[, "edf"]),
    deviance          = as.numeric(m_no_select$deviance),
    reml_score        = as.numeric(m_no_select$gcv.ubre),
    smooths           = sm_penalties(m_no_select),
    coef_split        = coef_per_smooth(m_no_select),
    fitted_terms      = fitted_terms(m_no_select, df)
  ),
  # The select=TRUE fit — the FEATURE under test
  select = list(
    sp                = sp_select,             # length 4 — 2 per smooth
    log_sp            = log(sp_select),
    sp_structure_note = "sp = [S1_orig, S1_null, S2_orig, S2_null]; null-space sp grows for unused terms",
    coefficients      = as.numeric(coef(m_select)),
    coef_names        = names(coef(m_select)),
    edf               = as.numeric(m_select$edf),
    edf_per_smooth    = as.numeric(summary(m_select)$s.table[, "edf"]),
    deviance          = as.numeric(m_select$deviance),
    reml_score        = as.numeric(m_select$gcv.ubre),
    smooths           = sm_penalties(m_select),  # each smooth now has S of length 2
    coef_split        = coef_per_smooth(m_select),
    fitted_terms      = fitted_terms(m_select, df)
  )
)

out_path <- "/home/alex/vibe_coding/nn_exploring/test_data/select_parity_basic.json"
write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE,
           digits = NA, matrix = "rowmajor", na = "null")
cat(sprintf("\nWrote %s\n", out_path))

cat("\n== per-smooth coef L2 norms ==\n")
for (j in seq_along(m_no_select$smooth)) {
  rng <- m_no_select$smooth[[j]]$first.para:m_no_select$smooth[[j]]$last.para
  l2_ns <- sqrt(sum(coef(m_no_select)[rng]^2))
  l2_s  <- sqrt(sum(coef(m_select)[rng]^2))
  cat(sprintf("  smooth %d (%s): no-select |β| = %.5g | select |β| = %.5g\n",
              j, m_no_select$smooth[[j]]$term, l2_ns, l2_s))
}

cat("\n== per-smooth edf ==\n")
edf_ns <- summary(m_no_select)$s.table[, "edf"]
edf_s  <- summary(m_select)$s.table[, "edf"]
for (j in seq_along(edf_ns)) {
  cat(sprintf("  smooth %d: no-select edf = %.4f | select edf = %.4f\n",
              j, edf_ns[j], edf_s[j]))
}
