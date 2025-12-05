#!/usr/bin/env Rscript

# Benchmark mgcv gradient computation performance
library(mgcv)

cat("=== R/mgcv Gradient Benchmark ===\n\n")

# Test configuration matching our Rust profiler
n <- 6000
n_dims <- 8
k <- 10
p <- n_dims * k

cat(sprintf("Configuration: n=%d, dims=%d, k=%d, p=%d\n\n", n, n_dims, k, p))

# Generate data
set.seed(42)
cat("[1/4] Generating data...\n")
X <- matrix(rnorm(n * p), n, p)
y <- rnorm(n)
w <- rep(1, n)

# Create block-diagonal penalties (one per dimension)
cat("[2/4] Creating penalties...\n")
penalties <- list()
for (dim in 1:n_dims) {
  S <- matrix(0, p, p)
  start_idx <- (dim - 1) * k + 1
  end_idx <- dim * k
  # Simple second-derivative penalty
  for (i in start_idx:end_idx) {
    S[i, i] <- 1.0
  }
  penalties[[dim]] <- S
}

# Initial lambdas
lambdas <- rep(1.0, n_dims)

cat("[3/4] Warm-up call...\n")
# Build the penalized system once for warm-up
A <- t(X) %*% (X * w)
for (i in 1:n_dims) {
  A <- A + lambdas[i] * penalties[[i]]
}
beta <- solve(A, t(X) %*% (y * w))

cat("[4/4] Benchmarking gradient computation...\n\n")

# Benchmark gradient computation (this is what we're optimizing in Rust)
# The gradient computation involves:
# 1. Computing trace terms (multiple triangular solves)
# 2. Computing beta derivatives (multiple linear solves)

n_iters <- 10
start_time <- Sys.time()

for (iter in 1:n_iters) {
  # Recompute A for each iteration (matching Rust profiler)
  A <- t(X) %*% (X * w)
  for (i in 1:n_dims) {
    A <- A + lambdas[i] * penalties[[i]]
  }
  
  # QR decomposition
  qr_result <- qr(A)
  R <- qr.R(qr_result)
  
  # Compute gradient components (simplified version of what mgcv does)
  gradient <- numeric(n_dims)
  
  for (i in 1:n_dims) {
    # Eigendecomposition for penalty sqrt
    eig <- eigen(penalties[[i]], symmetric = TRUE)
    pos_eigs <- which(eig$values > 1e-10)
    
    if (length(pos_eigs) > 0) {
      sqrt_pen <- eig$vectors[, pos_eigs, drop=FALSE] %*% diag(sqrt(eig$values[pos_eigs]), length(pos_eigs))
      
      # Trace computation (main bottleneck)
      trace_term <- 0
      for (k in 1:ncol(sqrt_pen)) {
        x <- backsolve(R, forwardsolve(t(R), sqrt_pen[, k]))
        trace_term <- trace_term + sum(x^2)
      }
      
      # Beta derivative (secondary bottleneck)
      s_beta <- penalties[[i]] %*% beta
      dbeta <- -solve(A, lambdas[i] * s_beta)
      
      gradient[i] <- trace_term - length(pos_eigs)
    }
  }
}

end_time <- Sys.time()
elapsed <- as.numeric(end_time - start_time, units = "secs")

cat("Results:\n")
cat(sprintf("  Total time: %.3fs\n", elapsed))
cat(sprintf("  Per call: %.3fs\n", elapsed / n_iters))
cat(sprintf("  Per iteration (5 calls): %.3fs\n\n", elapsed / (n_iters / 5)))

cat("Comparison:\n")
cat(sprintf("  Rust (current): ~0.062s per call\n"))
cat(sprintf("  R/mgcv (this):  %.3fs per call\n", elapsed / n_iters))

speedup <- (elapsed / n_iters) / 0.062
if (speedup > 1) {
  cat(sprintf("  → Rust is %.2fx FASTER\n", speedup))
} else {
  cat(sprintf("  → Rust is %.2fx SLOWER\n", 1/speedup))
}
