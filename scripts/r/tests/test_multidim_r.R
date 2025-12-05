# Test multi-dimensional convergence in R to compare with Rust

library(mgcv)

set.seed(123)
n <- 500
k <- 15

x1 <- runif(n)
x2 <- runif(n)
y <- sin(2 * pi * x1) + cos(2 * pi * x2) + rnorm(n, 0, 0.3)

cat("=== Testing d=2 convergence in R ===\n\n")

# Test with gam
cat("1. gam(method='REML'):\n")
m_gam <- gam(y ~ s(x1, k=k, bs='cr') + s(x2, k=k, bs='cr'), method="REML")
cat("   λ₁:", m_gam$sp[1], "\n")
cat("   λ₂:", m_gam$sp[2], "\n")
cat("   Iterations:", m_gam$outer.info$iter, "\n")
cat("   Gradient norm:", max(abs(m_gam$outer.info$grad)), "\n")
cat("   REML:", m_gam$gcv.ubre, "\n\n")

# Test with bam
cat("2. bam(method='REML'):\n")
m_bam <- bam(y ~ s(x1, k=k, bs='cr') + s(x2, k=k, bs='cr'), method="REML")
cat("   λ₁:", m_bam$sp[1], "\n")
cat("   λ₂:", m_bam$sp[2], "\n")
cat("   Iterations:", m_bam$outer.info$iter, "\n")
cat("   REML:", m_bam$gcv.ubre, "\n\n")

cat("3. Rust result for comparison:\n")
cat("   λ₁: 35.071968\n")
cat("   λ₂: 42.481781\n")
cat("   Gradient L∞: 0.068\n")
cat("   REML: 112.061347\n")
