#!/usr/bin/env Rscript
# Benchmark multi-dimensional GAM with R's mgcv

library(mgcv)

generate_multidim_data <- function(n, n_dims, seed=42) {
  set.seed(seed)

  # Generate predictors in [0, 1]
  X <- matrix(runif(n * n_dims), nrow=n, ncol=n_dims)

  # Generate true function
  y <- numeric(n)

  # Main effects
  if (n_dims >= 1) {
    y <- y + sin(2 * pi * X[,1])
  }
  if (n_dims >= 2) {
    y <- y + 0.5 * cos(3 * pi * X[,2])
  }
  if (n_dims >= 3) {
    y <- y + 0.3 * (X[,3]^2)
  }
  if (n_dims >= 4) {
    y <- y + 0.2 * exp(-5 * (X[,4] - 0.5)^2)
  }

  # Smaller contributions from remaining dimensions
  if (n_dims > 4) {
    for (i in 5:n_dims) {
      y <- y + 0.1 * sin(pi * X[,i])
    }
  }

  # Add noise
  y <- y + rnorm(n, 0, 0.2)

  list(X=X, y=y)
}

benchmark_r_gam <- function(X, y, k, n_runs=3) {
  n <- nrow(X)
  n_dims <- ncol(X)

  times <- numeric(n_runs)
  lambdas_list <- list()
  iterations <- numeric(n_runs)

  # Build formula
  formula_parts <- c("y ~ ")
  for (i in 1:n_dims) {
    if (i > 1) {
      formula_parts <- c(formula_parts, " + ")
    }
    formula_parts <- c(formula_parts, sprintf("s(x%d, bs='cr', k=%d)", i, k))
  }
  formula_str <- paste(formula_parts, collapse="")
  formula_obj <- as.formula(formula_str)

  for (run in 1:n_runs) {
    # Create data frame
    df <- data.frame(y=y)
    for (i in 1:n_dims) {
      df[[sprintf("x%d", i)]] <- X[,i]
    }

    # Fit model
    start_time <- Sys.time()
    fit <- gam(formula_obj, data=df, method='REML')
    end_time <- Sys.time()

    elapsed <- as.numeric(end_time - start_time, units="secs")
    times[run] <- elapsed
    lambdas_list[[run]] <- fit$sp
    iterations[run] <- fit$outer.info$iter

    cat(sprintf("  Run %d: %.4fs, λ=[%.4f %.4f %.4f]..., iter=%d\n",
                run, elapsed, fit$sp[1], fit$sp[2], fit$sp[3], fit$outer.info$iter))
  }

  list(
    mean_time = mean(times),
    std_time = sd(times),
    lambdas = colMeans(do.call(rbind, lambdas_list)),
    iterations = mean(iterations),
    times = times
  )
}

cat("================================================================================\n")
cat("R MGCV MULTI-DIMENSIONAL GAM BENCHMARK\n")
cat("================================================================================\n\n")

# Test configurations
configs <- list(
  list(n=6000, n_dims=8, k=10),
  list(n=6000, n_dims=10, k=10)
)

for (config in configs) {
  n <- config$n
  n_dims <- config$n_dims
  k <- config$k

  cat(sprintf("Configuration: n=%d, dimensions=%d, k=%d\n", n, n_dims, k))
  cat("--------------------------------------------------------------------------------\n")

  # Generate data
  cat(sprintf("Generating %d-dimensional data...\n", n_dims))
  data <- generate_multidim_data(n, n_dims)
  cat(sprintf("  Data shape: X=(%d, %d), y=(%d)\n\n", nrow(data$X), ncol(data$X), length(data$y)))

  # Benchmark
  cat("Benchmarking R's mgcv (3 runs)...\n")
  result <- benchmark_r_gam(data$X, data$y, k, n_runs=3)
  cat(sprintf("  Mean: %.4fs ± %.4fs\n", result$mean_time, result$std_time))
  cat(sprintf("  Lambdas: [%s]\n", paste(sprintf("%.8f", result$lambdas), collapse=" ")))
  cat(sprintf("  Iterations: %.1f\n", result$iterations))
  cat(sprintf("  Time per iteration: %.1fms\n\n", result$mean_time / result$iterations * 1000))

  cat("================================================================================\n\n")
}
