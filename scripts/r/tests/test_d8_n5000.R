# Test d=8, n=5000 with gam and bam

library(mgcv)

set.seed(123)
n <- 5000
d <- 8
k <- 8

cat("=== Testing n=5000, d=8, k=8 ===\n\n")

# Generate data
X <- matrix(runif(n * d), n, d)
y <- rowSums(sin(2 * pi * X)) + rnorm(n, 0, 0.3)

# Build formula
formula_parts <- paste0("s(X[,", 1:d, "], k=", k, ", bs='cr')", collapse=" + ")
formula_str <- paste("y ~", formula_parts)

cat("Data generated: n =", n, ", d =", d, ", k =", k, "\n\n")

# Test with gam
cat("1. gam(method='REML'):\n")
start_time <- Sys.time()
m_gam <- gam(as.formula(formula_str), method="REML")
gam_time <- as.numeric(difftime(Sys.time(), start_time, units="secs")) * 1000

cat("   Iterations:", m_gam$outer.info$iter, "\n")
cat("   Lambda (mean):", mean(m_gam$sp), "\n")
cat("   Lambda (min):", min(m_gam$sp), "\n")
cat("   Lambda (max):", max(m_gam$sp), "\n")
cat("   Gradient norm:", max(abs(m_gam$outer.info$grad)), "\n")
cat("   Time:", sprintf("%.1f ms", gam_time), "\n")
cat("   REML:", m_gam$gcv.ubre, "\n\n")

# Test with bam
cat("2. bam(method='REML'):\n")
start_time <- Sys.time()
m_bam <- bam(as.formula(formula_str), method="REML")
bam_time <- as.numeric(difftime(Sys.time(), start_time, units="secs")) * 1000

cat("   Iterations:", m_bam$outer.info$iter, "\n")
cat("   Lambda (mean):", mean(m_bam$sp), "\n")
cat("   Lambda (min):", min(m_bam$sp), "\n")
cat("   Lambda (max):", max(m_bam$sp), "\n")
cat("   Time:", sprintf("%.1f ms", bam_time), "\n")
cat("   REML:", m_bam$gcv.ubre, "\n\n")

cat("=== Summary ===\n")
cat(sprintf("%-15s %-10s %-15s %-15s\n", "Method", "Iterations", "Time (ms)", "Lambda (mean)"))
cat(sprintf("%-15s %-10d %-15.1f %-15.6f\n", "gam(REML)", m_gam$outer.info$iter, gam_time, mean(m_gam$sp)))
cat(sprintf("%-15s %-10d %-15.1f %-15.6f\n", "bam(REML)", m_bam$outer.info$iter, bam_time, mean(m_bam$sp)))
