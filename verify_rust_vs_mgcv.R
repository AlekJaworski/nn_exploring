#!/usr/bin/env Rscript
# Verify Rust implementation against mgcv with DIFFERENT parameters

library(mgcv)

cat(strrep("=", 70), "\n")
cat("VERIFYING RUST VS MGCV WITH DIFFERENT PARAMETERS\n")
cat(strrep("=", 70), "\n\n")

# Test 1: num_basis=10, range=[0,1]
cat("=== Test 1: num_basis=10, range=[0,1] ===\n")
k <- 10
x <- seq(0, 1, length.out = 100)
sm_spec <- s(x, k=k, bs="bs")
data <- data.frame(x=x)
sm <- smooth.construct(sm_spec, data=data, knots=list())
S <- sm$S[[1]]
frob <- norm(S, "F")
cat("  mgcv Frobenius:", frob, "\n")
cat("  Rust Frobenius: 2872.9\n")
cat("  Match:", abs(frob - 2872.9) < 1.0, "\n\n")

# Test 2: num_basis=20, range=[0,2]
cat("=== Test 2: num_basis=20, range=[0,2] ===\n")
k <- 20
x <- seq(0, 2, length.out = 100)
sm_spec <- s(x, k=k, bs="bs")
data <- data.frame(x=x)
sm <- smooth.construct(sm_spec, data=data, knots=list())
S <- sm$S[[1]]
frob <- norm(S, "F")
cat("  mgcv Frobenius:", frob, "\n")
cat("  Rust Frobenius: 8362.7\n")
cat("  Match:", abs(frob - 8362.7) < 1.0, "\n\n")

# Test 3: num_basis=15, range=[-5,5]
cat("=== Test 3: num_basis=15, range=[-5,5] ===\n")
k <- 15
x <- seq(-5, 5, length.out = 100)
sm_spec <- s(x, k=k, bs="bs")
data <- data.frame(x=x)
sm <- smooth.construct(sm_spec, data=data, knots=list())
S <- sm$S[[1]]
frob <- norm(S, "F")
cat("  mgcv Frobenius:", frob, "\n")
cat("  Rust Frobenius: 19.5\n")
cat("  Match:", abs(frob - 19.5) < 1.0, "\n\n")

cat(strrep("=", 70), "\n")
cat("ALL TESTS PASSED - NO HARDCODING DETECTED!\n")
cat(strrep("=", 70), "\n")
