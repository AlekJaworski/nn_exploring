#!/usr/bin/env Rscript
#
# Generate parity fixtures by running mgcv::gam on every input stub
# under tests/parity/_stubs/ and writing the full fixture to
# tests/parity/fixtures/.
#
# The Python build_input_stubs.py owns data generation (so numpy is the
# single rng source). This script just runs the model and emits the
# numbers a parity test will assert against.
#
# Run from repo root:
#   Rscript scripts/r/generate_parity_fixtures.R
#
# Or via Make:
#   make parity-fixtures
#
# Requires: mgcv, jsonlite

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

REPO_ROOT  <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), "..", ".."), mustWork = FALSE)
if (!dir.exists(file.path(REPO_ROOT, "tests", "parity"))) {
  # When sys.frame trick fails (e.g. piped Rscript), fall back to cwd.
  REPO_ROOT <- getwd()
}
STUBS_DIR    <- file.path(REPO_ROOT, "tests", "parity", "_stubs")
FIXTURES_DIR <- file.path(REPO_ROOT, "tests", "parity", "fixtures")

if (!dir.exists(STUBS_DIR)) {
  stop("No stubs at ", STUBS_DIR,
       ". Run python tests/parity/build_input_stubs.py first.")
}
dir.create(FIXTURES_DIR, showWarnings = FALSE, recursive = TRUE)

# ----- helpers -------------------------------------------------------- #

build_r_family <- function(family, link) {
  # mgcv accepts strings for some, but `Gamma` needs the function form.
  switch(
    family,
    "gaussian" = gaussian(link = link),
    "binomial" = binomial(link = link),
    "poisson"  = poisson(link = link),
    "Gamma"    = Gamma(link = link),
    "gamma"    = Gamma(link = link),
    stop("unknown family: ", family)
  )
}

build_formula <- function(d, k, bs) {
  # Always y ~ s(x0, bs="cr", k=10) + s(x1, bs="cr", k=15) + ...
  terms <- vapply(seq_len(d), function(i) {
    sprintf('s(x%d, bs="%s", k=%d)', i - 1, bs[[i]], k[[i]])
  }, character(1))
  as.formula(paste("y ~", paste(terms, collapse = " + ")))
}

# Strip mgcv smooth label `s(x0)` → key `x0`. Falls back to the raw label.
edf_key <- function(label) {
  m <- regmatches(label, regexpr("(?<=s\\()[^,)]+", label, perl = TRUE))
  if (length(m) == 0L) label else m
}

build_dataframe <- function(x_train, y_train, d) {
  df <- as.data.frame(x_train)
  names(df) <- paste0("x", seq_len(d) - 1L)
  df$y <- y_train
  df
}

predict_response <- function(fit, newdata) {
  unname(predict(fit, newdata = newdata, type = "response"))
}

predict_response_se <- function(fit, newdata) {
  pr <- predict(fit, newdata = newdata, type = "response", se.fit = TRUE)
  unname(pr$se.fit)
}

# ----- per-stub fitter ------------------------------------------------ #

fit_one <- function(stub_path) {
  stub <- jsonlite::fromJSON(stub_path, simplifyVector = TRUE)
  inp  <- stub$inputs

  # jsonlite returns nested-list arrays as matrices when the inner
  # length is constant — what we want. Defend against the 1-row corner.
  x_train  <- as.matrix(inp$x_train)
  if (ncol(x_train) != inp$d) x_train <- matrix(x_train, ncol = inp$d)
  x_test   <- matrix(unlist(inp$x_test),   ncol = inp$d, byrow = FALSE)
  x_extrap <- matrix(unlist(inp$x_extrap), ncol = inp$d, byrow = FALSE)

  # Ah — jsonlite already gives matrices for the nested arrays. Easier:
  if (is.list(inp$x_test))   x_test   <- do.call(rbind, inp$x_test)
  if (is.list(inp$x_extrap)) x_extrap <- do.call(rbind, inp$x_extrap)
  if (is.list(inp$x_train))  x_train  <- do.call(rbind, inp$x_train)

  d <- inp$d
  df_train  <- build_dataframe(x_train,  inp$y_train, d)
  df_test   <- as.data.frame(x_test);   names(df_test)   <- paste0("x", seq_len(d) - 1L)
  df_extrap <- as.data.frame(x_extrap); names(df_extrap) <- paste0("x", seq_len(d) - 1L)

  fam     <- build_r_family(inp$family, inp$link)
  formula <- build_formula(d, as.list(inp$k), as.list(inp$bs))

  weights <- if (is.null(inp$weights)) NULL else as.numeric(inp$weights)

  fit <- gam(formula, family = fam, data = df_train, method = inp$method,
             weights = weights)

  # ---- pull outputs ---------------------------------------------------
  beta <- unname(coef(fit))
  V    <- unname(vcov(fit))                         # Bayesian (Vp) by default
  lam  <- unname(fit$sp)                            # smoothing params

  edf_per_smooth <- list()
  for (sm in fit$smooth) {
    key  <- edf_key(sm$label)
    span <- sm$first.para:sm$last.para
    edf_per_smooth[[key]] <- sum(fit$edf[span])
  }

  preds_train <- predict_response(fit, df_train)
  preds_test  <- predict_response(fit, df_test)
  preds_xtrap <- predict_response(fit, df_extrap)

  preds_train_se <- predict_response_se(fit, df_train)
  preds_test_se  <- predict_response_se(fit, df_test)

  scale <- if (!is.null(fit$scale)) fit$scale else fit$sig2
  if (is.null(scale)) scale <- summary(fit)$scale

  # n_iter: gam stores at $iter (PiRLS) and $outer.info$iter (outer)
  n_iter <- if (!is.null(fit$outer.info$iter)) fit$outer.info$iter
            else if (!is.null(fit$iter))      fit$iter
            else NA_integer_

  list(
    schema_version = 1L,
    name           = stub$name,
    description    = stub$description,
    metadata       = list(
      mgcv_version = as.character(packageVersion("mgcv")),
      r_version    = R.version.string,
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    ),
    inputs         = inp,
    mgcv_output    = list(
      beta                 = beta,
      vcov                 = V,
      lambda               = lam,
      edf_per_smooth       = edf_per_smooth,
      edf_total            = sum(fit$edf),
      deviance             = as.numeric(deviance(fit)),
      scale                = as.numeric(scale),
      n_iter               = as.integer(n_iter),
      predictions_train    = as.numeric(preds_train),
      predictions_test     = as.numeric(preds_test),
      predictions_extrap   = as.numeric(preds_xtrap),
      predictions_train_se = as.numeric(preds_train_se),
      predictions_test_se  = as.numeric(preds_test_se)
    )
  )
}

# Tiny helper a la rlang::%||%
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ----- main loop ------------------------------------------------------ #

main <- function() {
  stubs <- list.files(STUBS_DIR, pattern = "\\.json$", full.names = TRUE)
  if (length(stubs) == 0L) {
    stop("No stubs in ", STUBS_DIR)
  }
  cat(sprintf("Generating %d parity fixtures...\n", length(stubs)))

  successes <- 0L
  failures  <- character()

  for (stub_path in stubs) {
    name <- sub("\\.json$", "", basename(stub_path))
    cat(sprintf("  %s ... ", name))
    res <- tryCatch({
      fixture <- fit_one(stub_path)
      out_path <- file.path(FIXTURES_DIR, paste0(name, ".json"))
      jsonlite::write_json(
        fixture, out_path,
        auto_unbox = TRUE, pretty = TRUE, digits = NA, matrix = "rowmajor",
        null = "null"
      )
      cat("OK\n")
      "ok"
    }, error = function(e) {
      cat("FAILED:", conditionMessage(e), "\n")
      "fail"
    })
    if (identical(res, "ok")) successes <- successes + 1L
    else failures <- c(failures, name)
  }

  cat(sprintf("\n%d/%d fixtures generated.\n", successes, length(stubs)))
  if (length(failures) > 0L) {
    cat("Failures:\n"); cat(paste0("  ", failures, "\n"), sep = "")
    quit(status = 1L)
  }
}

main()
