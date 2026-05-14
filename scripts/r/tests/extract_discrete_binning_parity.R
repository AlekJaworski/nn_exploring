#!/usr/bin/env Rscript
# extract_discrete_binning_parity.R
#
# Parity capture for mgcv's `bam(..., discrete = TRUE)` covariate-binning
# fast path vs the un-binned `gam(...)` baseline. Companion to
# `docs/DISCRETE_BINNING_DESIGN.md`.
#
# Captures two scenarios:
#   1. Synthetic, controlled ties (small n, predictable structure)
#   2. Production fixture `data/sale_price_fixtures/split_0_train.parquet`
#      — real real-estate-data tie distribution (monthly_index ~12 unique,
#      days_in_current_price_regime ~22 unique, cum_dom ~168 unique, etc.).
#
# For each scenario we fit:
#   * `m_gam`  — gam(...) reference (no binning, full n)
#   * `m_bam`  — bam(..., discrete = TRUE) (covariate binning + scatter-gather)
#
# And dump:
#   * Per-covariate bin assignments (which obs went into which bin) via
#     `mgcv:::discrete.mf` (the function bam calls internally before fitting).
#     Output: per-column `index` (length n), `n_bins`, sorted bin values.
#   * Bin counts (frequency table of the per-column index vector).
#   * Coefficients from both fits (expected ~1e-3 disagreement: binning is
#     an approximation; the design doc tolerance band).
#   * sp / log_sp / edf / deviance / REML score.
#   * Wall-clock time per fit (to corroborate the 30× gap in the perf memo).
#   * Per-column unique-value counts, for sanity checking the tie hypothesis.
#
# Outputs:
#   test_data/discrete_binning_parity_synthetic.json
#   test_data/discrete_binning_parity_production.json   (skipped if parquet unread)
#
# nanoparquet is used to read the production parquet — `install.packages("nanoparquet")`
# if it isn't already present.

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

# -------------------------------------------------------------------------
# Helper: capture all the per-scenario parity state given a fit pair.
# -------------------------------------------------------------------------

extract_fit <- function(m, tag) {
  list(
    tag              = tag,
    elapsed_sec      = NA_real_,   # filled by caller
    coefficients     = as.numeric(coef(m)),
    coef_names       = names(coef(m)),
    sp               = as.numeric(m$sp),
    log_sp           = as.numeric(log(m$sp)),
    edf_total        = as.numeric(sum(m$edf)),
    edf_per_smooth   = as.numeric(summary(m)$s.table[, "edf"]),
    deviance         = as.numeric(m$deviance),
    aic              = as.numeric(m$aic),
    reml_score       = as.numeric(m$gcv.ubre),
    converged        = m$converged,
    iter             = if (!is.null(m$iter)) m$iter else NA_integer_,
    fitted_first10   = as.numeric(head(m$fitted.values, 10)),
    optimizer        = if (!is.null(m$optimizer)) m$optimizer else NA_character_
  )
}

# Capture mgcv's internal discretized state for a bam fit.
# This is the load-bearing intermediate-state extraction the design doc
# needs to mirror in Rust. The four key data structures are:
#
#   G$kd  -- n × n_marg integer matrix of per-row bin indices (1-based; one
#            column per "marginal" — a smooth.spec's covariate(s)).
#   G$ks  -- n_marg × 2 matrix mapping each marginal to its column range
#            inside kd (so we can find which column(s) belong to smooth i).
#   nr    -- per-marginal vector of "number of unique bin values".
#   Xd    -- list of m_marg × k_basis basis matrices, one per marginal.
#
# These four pieces collectively define mgcv's discretized design. Predicting
# eta at observation i requires:
#   eta[i] = sum over smooths j: Xd[j][kd[i, kj], :] @ beta_j
# This is exactly what mgcv's `Xbd` C routine does.
capture_discrete_state <- function(m_bam) {
  if (is.null(m_bam$dinfo) || !isTRUE(m_bam$discretize) && !exists("Xd", where = m_bam)) {
    # bam refit with discrete=TRUE always sets $discretize but on some paths
    # it can be in $G or stashed differently. Be lenient.
  }
  # mgcv stashes the discretized state in different attributes depending on
  # version; check a few candidate locations.
  G_disc <- m_bam$G
  if (is.null(G_disc)) G_disc <- m_bam
  out <- list()
  out$has_kd <- !is.null(G_disc$kd)
  out$has_Xd <- !is.null(G_disc$Xd)
  if (!is.null(G_disc$kd)) {
    # kd is n × n_marg; large for production data, so subsample first
    # 100 rows for the JSON dump (full thing is huge).
    kd <- G_disc$kd
    out$kd_dim       <- dim(kd)
    out$kd_first100  <- if (nrow(kd) >= 100) kd[1:100, , drop = FALSE] else kd
    out$nr           <- as.integer(G_disc$nr)
    if (!is.null(G_disc$ks)) {
      out$ks <- G_disc$ks
    }
  }
  if (!is.null(G_disc$Xd)) {
    out$Xd_dims <- lapply(G_disc$Xd, dim)
    # Don't dump full Xd matrices into JSON — keep just shapes and a
    # spot-check (top-left 3x3 of each).
    out$Xd_topleft <- lapply(G_disc$Xd, function(M) {
      n_top <- min(3, nrow(M))
      p_top <- min(3, ncol(M))
      M[1:n_top, 1:p_top, drop = FALSE]
    })
  }
  out
}

run_pair <- function(formula, data, family, method = "REML",
                     bam_method = "fREML", select = FALSE,
                     weights = NULL, scenario_name = "") {
  cat(sprintf("\n== Scenario: %s (n=%d) ==\n", scenario_name, nrow(data)))
  # gam fit (reference)
  t1 <- system.time({
    m_gam <- if (is.null(weights)) {
      gam(formula, data = data, family = family, method = method,
          select = select)
    } else {
      gam(formula, data = data, family = family, method = method,
          select = select, weights = weights)
    }
  })
  cat(sprintf("  gam(..., method=%s):                    %.3fs\n",
              method, t1["elapsed"]))

  # bam fit, discrete = FALSE — purely for apples-to-apples discrete vs not.
  t2 <- system.time({
    m_bam_nondisc <- if (is.null(weights)) {
      bam(formula, data = data, family = family, method = bam_method,
          select = select, discrete = FALSE)
    } else {
      bam(formula, data = data, family = family, method = bam_method,
          select = select, discrete = FALSE, weights = weights)
    }
  })
  cat(sprintf("  bam(..., discrete=FALSE):              %.3fs\n",
              t2["elapsed"]))

  # bam fit, discrete = TRUE — the path we want to port.
  t3 <- system.time({
    m_bam_disc <- if (is.null(weights)) {
      bam(formula, data = data, family = family, method = bam_method,
          select = select, discrete = TRUE)
    } else {
      bam(formula, data = data, family = family, method = bam_method,
          select = select, discrete = TRUE, weights = weights)
    }
  })
  cat(sprintf("  bam(..., discrete=TRUE):               %.3fs   <-- target perf\n",
              t3["elapsed"]))

  # Pretty coef-distance summary (the binning bias the design doc warns about)
  c_gam <- as.numeric(coef(m_gam))
  c_bam_disc <- as.numeric(coef(m_bam_disc))
  # Names should match; if any mismatch, just skip the comparison.
  if (length(c_gam) == length(c_bam_disc)) {
    coef_max_abs <- max(abs(c_gam - c_bam_disc))
    coef_rel     <- coef_max_abs / max(1e-12, max(abs(c_gam)))
    cat(sprintf("  coef max |gam - bam_disc| = %.3e  (rel %.3e)\n",
                coef_max_abs, coef_rel))
  }

  fit_gam      <- extract_fit(m_gam,      "gam_reference")
  fit_gam$elapsed_sec      <- as.numeric(t1["elapsed"])
  fit_bam_full <- extract_fit(m_bam_nondisc, "bam_nondiscrete")
  fit_bam_full$elapsed_sec <- as.numeric(t2["elapsed"])
  fit_bam_disc <- extract_fit(m_bam_disc, "bam_discrete")
  fit_bam_disc$elapsed_sec <- as.numeric(t3["elapsed"])

  list(
    scenario        = scenario_name,
    n               = nrow(data),
    formula         = deparse(formula),
    family_str      = if (is.character(family)) family else family$family,
    method          = method,
    bam_method      = bam_method,
    select          = select,
    weighted        = !is.null(weights),
    fit_gam            = fit_gam,
    fit_bam_nondiscrete = fit_bam_full,
    fit_bam_discrete    = fit_bam_disc,
    perf_ratio_discrete_vs_gam = as.numeric(t1["elapsed"]) /
                                  max(1e-9, as.numeric(t3["elapsed"])),
    discrete_state  = capture_discrete_state(m_bam_disc)
  )
}

# Capture the mgcv:::discrete.mf step in isolation — the actual binner that
# bam invokes under the hood. This exposes the per-covariate bin maps that a
# Rust port has to reproduce.
capture_discrete_mf <- function(formula, data, m = NULL) {
  # mgcv's bam() pre-processes the formula via interpret.gam then passes the
  # parsed gp + the model frame into discrete.mf. We do the same.
  gp <- mgcv:::interpret.gam(formula)
  mf <- model.frame(gp$fake.formula, data = data)
  pmf_names <- character(0)  # no parametric terms in the synthetic example
  if (!is.null(gp$pf) && length(attr(terms(gp$pf), "term.labels")) > 0) {
    pmf <- model.frame(gp$pf, data = data, na.action = na.pass)
    pmf_names <- names(pmf)
  }
  dk <- mgcv:::discrete.mf(gp, mf, pmf_names, m = m)
  # dk has $mf (compressed data — only the unique rows!), $k (n × n_marg
  # index matrix), $nr (per-marginal n_unique), $ks (n_marg × 2 column
  # ranges into k).
  list(
    n_obs        = nrow(mf),
    nr           = as.integer(dk$nr),
    ks           = dk$ks,
    k_dim        = dim(dk$k),
    k_first50    = if (nrow(dk$k) >= 50) dk$k[1:50, , drop = FALSE] else dk$k,
    compressed_mf_rows = nrow(dk$mf),
    # First 20 rows of the compressed model frame, so we can verify that
    # the bin midpoints/values match what a Rust port should generate.
    compressed_mf_head = head(dk$mf, 20),
    # Unique-value count per column (sanity check vs nr).
    nunique_per_col = sapply(dk$mf, function(x) length(unique(x)))
  )
}

# =========================================================================
# SCENARIO 1 — synthetic, small, designed to show binning at work.
# =========================================================================
set.seed(42)
n_syn <- 2000
# Three covariates with very different tie structures (mirroring production):
#   x1 — 5 unique values  (extreme ties, like monthly_index)
#   x2 — 50 unique values (moderate ties, like cum_dom)
#   x3 — fully unique     (no ties, baseline)
x1_synthetic <- sample(seq(0, 1, length.out = 5),  n_syn, replace = TRUE)
x2_synthetic <- sample(seq(0, 1, length.out = 50), n_syn, replace = TRUE)
x3_synthetic <- runif(n_syn)
y_synthetic <- 2 * sin(2 * pi * x1_synthetic) +
               cos(2 * pi * x2_synthetic) +
               0.5 * x3_synthetic +
               rnorm(n_syn, sd = 0.3)
df_syn <- data.frame(x1 = x1_synthetic, x2 = x2_synthetic,
                     x3 = x3_synthetic, y = y_synthetic)

result_synthetic <- run_pair(
  formula = y ~ s(x1, k = 5, bs = "cr") +
               s(x2, k = 10, bs = "cr") +
               s(x3, k = 10, bs = "cr"),
  data = df_syn, family = "gaussian",
  method = "REML", bam_method = "fREML",
  scenario_name = "synthetic_controlled_ties"
)
# Add the standalone discrete.mf capture (the bin-assignment step on its own).
result_synthetic$discrete_mf <- capture_discrete_mf(
  formula = y ~ s(x1, k = 5, bs = "cr") +
               s(x2, k = 10, bs = "cr") +
               s(x3, k = 10, bs = "cr"),
  data = df_syn
)

cat(sprintf("\nDiscrete.mf summary (synthetic):\n"))
cat(sprintf("  bin counts per marginal: %s\n",
            paste(result_synthetic$discrete_mf$nr, collapse = ", ")))
cat(sprintf("  compressed model frame: %d rows (vs n=%d)\n",
            result_synthetic$discrete_mf$compressed_mf_rows,
            result_synthetic$discrete_mf$n_obs))

out_syn <- "/home/alex/vibe_coding/nn_exploring/test_data/discrete_binning_parity_synthetic.json"
write_json(result_synthetic, out_syn, auto_unbox = TRUE, pretty = TRUE,
           digits = NA, matrix = "rowmajor", na = "null")
cat(sprintf("\nWrote %s\n", out_syn))

# =========================================================================
# SCENARIO 2 — production fixture (real ties from neighbourhoods sale-price)
# =========================================================================
fixture_path <- "/home/alex/vibe_coding/nn_exploring/data/sale_price_fixtures/split_0_train.parquet"

read_parquet_safely <- function(p) {
  if (requireNamespace("nanoparquet", quietly = TRUE)) {
    return(nanoparquet::read_parquet(p))
  }
  if (requireNamespace("arrow", quietly = TRUE)) {
    return(as.data.frame(arrow::read_parquet(p)))
  }
  NULL
}

df_prod <- read_parquet_safely(fixture_path)

if (is.null(df_prod)) {
  cat(sprintf("\n[skip] Could not read %s — install nanoparquet or arrow to enable\n",
              fixture_path))
} else {
  cat(sprintf("\nProduction fixture loaded: %d × %d\n",
              nrow(df_prod), ncol(df_prod)))
  smooths <- c("current_list_price", "price_change_pct_from_original",
               "cum_dom_before_current_regime", "days_in_current_price_regime",
               "monthly_index")
  # K per smooth: cap by nunique (mgcv-rust integration mirrors this).
  k_per <- sapply(smooths, function(c) min(7, length(unique(df_prod[[c]]))))
  k_per["monthly_index"] <- min(5, length(unique(df_prod$monthly_index)))
  cat("Per-smooth k caps:\n")
  for (i in seq_along(smooths)) cat(sprintf("  %s: k=%d (nunique=%d)\n",
                                            smooths[i], k_per[i],
                                            length(unique(df_prod[[smooths[i]]]))))

  # Build the formula string. mgcv evaluates each `s(...)` from the formula's
  # environment, so the embedded `k = ...` must look up cleanly.
  fparts <- sapply(seq_along(smooths), function(i) {
    sprintf("s(%s, k = %d, bs = \"cr\")", smooths[i], k_per[i])
  })
  formula_prod <- as.formula(paste("sale_to_list_price_ratio ~",
                                   paste(fparts, collapse = " + ")))

  # Use Gaussian here — the perf gap to capture is binning-vs-not, not
  # scat-vs-gaussian. The same binning machinery feeds scat in bam (proven
  # by the customer-feedback note's bam(scat, discrete=TRUE) timings).
  result_prod <- run_pair(
    formula = formula_prod,
    data    = df_prod,
    family  = "gaussian",
    method  = "REML",
    bam_method = "fREML",
    scenario_name = "production_sale_price_split0"
  )
  result_prod$discrete_mf <- capture_discrete_mf(
    formula = formula_prod, data = df_prod
  )

  cat(sprintf("\nDiscrete.mf summary (production):\n"))
  cat(sprintf("  bin counts per marginal: %s\n",
              paste(result_prod$discrete_mf$nr, collapse = ", ")))
  cat(sprintf("  compressed model frame: %d rows (vs n=%d)\n",
              result_prod$discrete_mf$compressed_mf_rows,
              result_prod$discrete_mf$n_obs))

  # Also try a scat fit — the original perf complaint. Discrete bam(scat)
  # is what the user benchmarked at 155-272ms vs 5-13s for gam(scat).
  cat("\n-- scat parity (production) --\n")
  tryCatch({
    t_scat <- system.time({
      m_scat_gam <- gam(formula_prod, data = df_prod,
                       family = scat(), method = "REML")
    })
    cat(sprintf("  gam(scat, REML):            %.3fs\n", t_scat["elapsed"]))
    t_scat_bam <- system.time({
      m_scat_bam <- bam(formula_prod, data = df_prod,
                       family = scat(), method = "fREML", discrete = TRUE)
    })
    cat(sprintf("  bam(scat, fREML, disc=TRUE): %.3fs\n",
                t_scat_bam["elapsed"]))
    result_prod$scat_perf <- list(
      gam_scat_sec = as.numeric(t_scat["elapsed"]),
      bam_scat_discrete_sec = as.numeric(t_scat_bam["elapsed"]),
      ratio = as.numeric(t_scat["elapsed"]) / as.numeric(t_scat_bam["elapsed"]),
      reml_score_gam = m_scat_gam$gcv.ubre,
      reml_score_bam = m_scat_bam$gcv.ubre,
      coef_max_abs_diff = max(abs(coef(m_scat_gam) - coef(m_scat_bam)))
    )
  }, error = function(e) {
    cat(sprintf("  scat fit failed: %s\n", e$message))
  })

  out_prod <- "/home/alex/vibe_coding/nn_exploring/test_data/discrete_binning_parity_production.json"
  write_json(result_prod, out_prod, auto_unbox = TRUE, pretty = TRUE,
             digits = NA, matrix = "rowmajor", na = "null")
  cat(sprintf("\nWrote %s\n", out_prod))
}

cat("\n== Capture done ==\n")
