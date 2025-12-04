## Comprehensive benchmark: Rust vs R (gam/bam) across various problem sizes
library(mgcv)

run_benchmark <- function(n, d, k) {
  set.seed(123)

  # Generate data
  data_list <- lapply(1:d, function(i) runif(n))
  names(data_list) <- paste0("x", 1:d)

  # Create response from smooth functions
  y <- Reduce(`+`, lapply(data_list, function(x) sin(2 * pi * x))) + rnorm(n, 0, 0.3)

  # Create data frame
  df <- as.data.frame(c(list(y = y), data_list))

  # Build formula
  smooth_terms <- paste0("s(x", 1:d, ", bs='cr', k=", k, ")", collapse = " + ")
  formula_str <- paste("y ~", smooth_terms)
  formula_obj <- as.formula(formula_str)

  # Benchmark gam()
  gam_time <- system.time({
    gam_fit <- gam(formula_obj, data = df, method = "REML")
  })["elapsed"] * 1000

  # Benchmark bam()
  bam_time <- system.time({
    bam_fit <- bam(formula_obj, data = df, method = "REML")
  })["elapsed"] * 1000

  # Extract smoothing parameters
  gam_sp <- gam_fit$sp
  bam_sp <- bam_fit$sp

  list(
    n = n, d = d, k = k, p = d * k,
    gam_time = gam_time,
    bam_time = bam_time,
    gam_sp_mean = mean(gam_sp),
    bam_sp_mean = mean(bam_sp)
  )
}

cat("\n=== Comprehensive GAM Benchmark (R baseline) ===\n\n")
cat("Testing gam() and bam() across various problem sizes\n\n")

# Test configurations
configs <- list(
  list(n=1000, d=1, k=10),
  list(n=1000, d=2, k=10),
  list(n=1000, d=4, k=10),

  list(n=2000, d=1, k=10),
  list(n=2000, d=2, k=10),
  list(n=2000, d=4, k=10),
  list(n=2000, d=8, k=8),

  list(n=5000, d=1, k=10),
  list(n=5000, d=2, k=10),
  list(n=5000, d=4, k=8),
  list(n=5000, d=8, k=8),

  list(n=10000, d=1, k=10),
  list(n=10000, d=2, k=10),
  list(n=10000, d=4, k=8)
)

results <- list()
for (cfg in configs) {
  cat(sprintf("Testing n=%5d, d=%d, k=%2d... ", cfg$n, cfg$d, cfg$k))
  flush.console()

  result <- run_benchmark(cfg$n, cfg$d, cfg$k)
  results <- c(results, list(result))

  cat(sprintf("gam: %6.1fms, bam: %6.1fms\n", result$gam_time, result$bam_time))
}

# Print summary table
cat("\n┌─────────┬─────┬─────┬────────┬──────────────┬──────────────┬─────────────┐\n")
cat("│    n    │  d  │  k  │  p=d×k │  gam (ms)    │  bam (ms)    │ gam/bam     │\n")
cat("├─────────┼─────┼─────┼────────┼──────────────┼──────────────┼─────────────┤\n")

for (r in results) {
  ratio <- r$gam_time / r$bam_time
  cat(sprintf("│ %7d │ %3d │ %3d │ %6d │ %12.1f │ %12.1f │ %11.2f │\n",
              r$n, r$d, r$k, r$p, r$gam_time, r$bam_time, ratio))
}

cat("└─────────┴─────┴─────┴────────┴──────────────┴──────────────┴─────────────┘\n")
cat("\nNotes:\n")
cat("- All using REML optimization\n")
cat("- Cubic regression splines (bs='cr')\n")
cat("- Single-run measurements (not averaged)\n")
