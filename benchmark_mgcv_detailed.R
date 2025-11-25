#!/usr/bin/env Rscript

# Detailed timing breakdown for mgcv
library(mgcv)

n <- 6000
n_dims <- 8
k <- 10
p <- n_dims * k

set.seed(42)
X <- matrix(rnorm(n * p), n, p)
y <- rnorm(n)
w <- rep(1, n)

penalties <- list()
for (dim in 1:n_dims) {
  S <- matrix(0, p, p)
  start_idx <- (dim - 1) * k + 1
  end_idx <- dim * k
  for (i in start_idx:end_idx) {
    S[i, i] <- 1.0
  }
  penalties[[dim]] <- S
}

lambdas <- rep(1.0, n_dims)

cat("=== Detailed R Timing Breakdown ===\n\n")

# Time eigendecomposition (penalty sqrt)
cat("[1/4] Penalty square roots (eigendecomp)...\n")
start <- Sys.time()
sqrt_penalties <- list()
for (i in 1:n_dims) {
  eig <- eigen(penalties[[i]], symmetric = TRUE)
  pos_eigs <- which(eig$values > 1e-10)
  sqrt_pen <- eig$vectors[, pos_eigs, drop=FALSE] %*% diag(sqrt(eig$values[pos_eigs]), length(pos_eigs))
  sqrt_penalties[[i]] <- sqrt_pen
}
eig_time <- as.numeric(Sys.time() - start, units = "secs")
cat(sprintf("  Time: %.3fs\n\n", eig_time))

# Time QR decomposition
cat("[2/4] QR decomposition...\n")
A <- t(X) %*% (X * w)
for (i in 1:n_dims) {
  A <- A + lambdas[i] * penalties[[i]]
}
start <- Sys.time()
qr_result <- qr(A)
R_mat <- qr.R(qr_result)
qr_time <- as.numeric(Sys.time() - start, units = "secs")
cat(sprintf("  Time: %.3fs\n\n", qr_time))

# Time trace computations
cat("[3/4] Trace computations...\n")
start <- Sys.time()
for (i in 1:n_dims) {
  sqrt_pen <- sqrt_penalties[[i]]
  trace_term <- 0
  for (k in 1:ncol(sqrt_pen)) {
    x <- backsolve(R_mat, forwardsolve(t(R_mat), sqrt_pen[, k]))
    trace_term <- trace_term + sum(x^2)
  }
}
trace_time <- as.numeric(Sys.time() - start, units = "secs")
cat(sprintf("  Time: %.3fs\n\n", trace_time))

# Time beta derivatives
cat("[4/4] Beta derivatives...\n")
beta <- solve(A, t(X) %*% (y * w))
start <- Sys.time()
for (i in 1:n_dims) {
  s_beta <- penalties[[i]] %*% beta
  dbeta <- solve(A, lambdas[i] * s_beta)
}
beta_time <- as.numeric(Sys.time() - start, units = "secs")
cat(sprintf("  Time: %.3fs\n\n", beta_time))

total <- eig_time + qr_time + trace_time + beta_time

cat("=== R Breakdown ===\n")
cat(sprintf("  Eigendecomp:    %.3fs (%.1f%%)\n", eig_time, 100 * eig_time / total))
cat(sprintf("  QR:             %.3fs (%.1f%%)\n", qr_time, 100 * qr_time / total))
cat(sprintf("  Trace:          %.3fs (%.1f%%)\n", trace_time, 100 * trace_time / total))
cat(sprintf("  Beta derivs:    %.3fs (%.1f%%)\n", beta_time, 100 * beta_time / total))
cat("  ─────────────────────────\n")
cat(sprintf("  Total:          %.3fs\n\n", total))

cat("Comparison to Rust:\n")
cat(sprintf("  Rust eigendecomp: ~0.011s (R: %.3fs)\n", eig_time))
cat(sprintf("  Rust QR:          ~0.054s (R: %.3fs)\n", qr_time))
cat(sprintf("  Rust trace:       ~0.001s (R: %.3fs)\n", trace_time))
cat(sprintf("  Rust beta:        ~0.000s (R: %.3fs)\n", beta_time))
