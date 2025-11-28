# Comprehensive performance comparison: bam vs gam vs Rust Newton
# Testing different n (sample size), d (dimensions), k (basis size)

library(mgcv)

set.seed(123)

results <- data.frame(
  n = integer(),
  d = integer(),
  k = integer(),
  method = character(),
  iterations = integer(),
  lambda = numeric(),
  time_ms = numeric(),
  stringsAsFactors = FALSE
)

# Test configurations: (n, d, k)
configs <- list(
  c(100, 1, 10),
  c(500, 1, 20),
  c(1000, 1, 20),
  c(2000, 1, 30),
  c(500, 2, 15),
  c(1000, 2, 15),
  c(500, 3, 12),
  c(5000, 1, 30),
  c(10000, 1, 30)
)

cat("=== Performance Comparison: bam() vs gam() ===\n\n")
cat(sprintf("%-8s %-6s %-6s %-20s %-10s %-15s %-10s\n",
            "n", "d", "k", "Method", "Iterations", "Lambda", "Time(ms)"))
cat(strrep("-", 85), "\n")

for (config in configs) {
  n <- config[1]
  d <- config[2]
  k <- config[3]

  # Generate data
  if (d == 1) {
    x <- runif(n)
    y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)
    formula_str <- sprintf("y ~ s(x, k=%d, bs='cr')", k)
  } else if (d == 2) {
    x1 <- runif(n)
    x2 <- runif(n)
    y <- sin(2 * pi * x1) + cos(2 * pi * x2) + rnorm(n, 0, 0.3)
    formula_str <- sprintf("y ~ s(x1, k=%d, bs='cr') + s(x2, k=%d, bs='cr')", k, k)
  } else if (d == 3) {
    x1 <- runif(n)
    x2 <- runif(n)
    x3 <- runif(n)
    y <- sin(2*pi*x1) + cos(2*pi*x2) + sin(4*pi*x3) + rnorm(n, 0, 0.3)
    formula_str <- sprintf("y ~ s(x1, k=%d, bs='cr') + s(x2, k=%d, bs='cr') + s(x3, k=%d, bs='cr')", k, k, k)
  }

  # Test gam() with Newton (default)
  tryCatch({
    start_time <- Sys.time()
    m_gam <- gam(as.formula(formula_str), method="REML")
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units="secs")) * 1000

    lambda_mean <- mean(m_gam$sp)
    iters <- m_gam$outer.info$iter

    cat(sprintf("%-8d %-6d %-6d %-20s %-10d %-15.4f %-10.1f\n",
                n, d, k, "gam(REML/Newton)", iters, lambda_mean, elapsed))

    results <- rbind(results, data.frame(
      n=n, d=d, k=k, method="gam(Newton)",
      iterations=iters, lambda=lambda_mean, time_ms=elapsed
    ))
  }, error = function(e) {
    cat(sprintf("%-8d %-6d %-6d %-20s %-10s %-15s %-10s\n",
                n, d, k, "gam(REML/Newton)", "FAILED", "-", "-"))
  })

  # Test bam() with Newton (default)
  tryCatch({
    start_time <- Sys.time()
    m_bam <- bam(as.formula(formula_str), method="REML")
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units="secs")) * 1000

    lambda_mean <- mean(m_bam$sp)
    iters <- m_bam$outer.info$iter

    cat(sprintf("%-8d %-6d %-6d %-20s %-10d %-15.4f %-10.1f\n",
                n, d, k, "bam(REML/Newton)", iters, lambda_mean, elapsed))

    results <- rbind(results, data.frame(
      n=n, d=d, k=k, method="bam(Newton)",
      iterations=iters, lambda=lambda_mean, time_ms=elapsed
    ))
  }, error = function(e) {
    cat(sprintf("%-8d %-6d %-6d %-20s %-10s %-15s %-10s\n",
                n, d, k, "bam(REML/Newton)", "FAILED", "-", "-"))
  })

  cat("\n")
}

# Save results
write.csv(results, "performance_comparison.csv", row.names=FALSE)

cat("\n=== Summary Statistics ===\n\n")
cat("Average iterations by method:\n")
agg <- aggregate(iterations ~ method, data=results, FUN=mean)
print(agg)

cat("\n\nAverage time by method (ms):\n")
agg_time <- aggregate(time_ms ~ method, data=results, FUN=mean)
print(agg_time)

cat("\n\nResults saved to performance_comparison.csv\n")
