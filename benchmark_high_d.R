# Benchmark d=4 and d=8 with bam/gam

library(mgcv)

set.seed(123)

test_config <- function(n, d, k) {
  cat(sprintf("\n=== Testing n=%d, d=%d, k=%d ===\n", n, d, k))
  
  # Generate data
  X <- matrix(runif(n * d), n, d)
  y <- rowSums(sin(2 * pi * X)) + rnorm(n, 0, 0.3)
  
  # Build formula
  formula_parts <- paste0("s(X[,", 1:d, "], k=", k, ", bs='cr')", collapse=" + ")
  formula_str <- paste("y ~", formula_parts)
  
  # Test with gam
  cat("gam(REML):\n")
  m_gam <- gam(as.formula(formula_str), method="REML")
  cat("  Iterations:", m_gam$outer.info$iter, "\n")
  cat("  Lambda (mean):", mean(m_gam$sp), "\n")
  cat("  Gradient norm:", max(abs(m_gam$outer.info$grad)), "\n\n")
  
  # Test with bam
  cat("bam(REML):\n")
  m_bam <- bam(as.formula(formula_str), method="REML")
  cat("  Iterations:", m_bam$outer.info$iter, "\n")
  cat("  Lambda (mean):", mean(m_bam$sp), "\n\n")
}

# Test configurations
test_config(500, 4, 10)
test_config(500, 8, 8)
test_config(1000, 4, 10)
test_config(1000, 8, 8)
