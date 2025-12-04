#!/usr/bin/env Rscript

library(mgcv)

set.seed(42)
n <- 500
k <- 20
x <- seq(0, 1, length.out = n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.1)

cat("Creating GAM with BS splines, k=20\n")
cat(strrep("=", 70), "\n\n")

# We need to manually create the smooth object to extract internals
# Create smooth specification
sm_spec <- s(x, k=k, bs="bs")

# Create smooth constructor
data <- data.frame(x=x)
knots <- list()

# Call smooth.construct
sm <- smooth.construct(sm_spec, data=data, knots=knots)

cat("Smooth object created\n")
cat("Number of coefficients:", sm$bs.dim, "\n")
cat("Knots (first 10):", sm$knots[1:10], "\n")
cat("Penalty matrix S[[1]] shape:", dim(sm$S[[1]]), "\n\n")

# Now extract the EXACT intermediate values
# Re-run the penalty construction code with debugging

# Get parameters
degree <- sm$p.order[1]  # m[1]
m2 <- sm$p.order[2]  # Second order derivative penalty
pord <- degree - m2

cat("Parameters:\n")
cat("  degree (m[1]):", degree, "\n")
cat("  m2 (penalty order):", m2, "\n")
cat("  pord:", pord, "\n\n")

# Extract interior knots k0
nk <- sm$bs.dim - degree + 1
k0 <- sm$knots[(degree + 1):(degree + nk)]

cat("Interior knots k0:\n")
cat("  Length:", length(k0), "\n")
cat("  First few:", k0[1:5], "\n")
cat("  Last few:", tail(k0, 5), "\n\n")

# Compute h
h <- diff(k0)
cat("Knot spacings h:\n")
cat("  Length:", length(h), "\n")
cat("  First value:", h[1], "\n")
cat("  All equal?", all(abs(h - h[1]) < 1e-10), "\n\n")

# Subdivide for pord > 0
h1 <- rep(h/pord, each = pord)
k1 <- cumsum(c(k0[1], h1))

cat("Evaluation points k1:\n")
cat("  Length:", length(k1), "\n")
cat("  First few:", k1[1:5], "\n")
cat("  Last few:", tail(k1, 5), "\n\n")

# Build W1 matrix
seq_vals <- seq(-1, 1, length = pord + 1)
cat("seq_vals:", seq_vals, "\n\n")

# Build Vandermonde-like matrix
powers_vec <- rep(seq_vals, pord + 1)^rep(0:pord, each = pord + 1)
powers_matrix <- matrix(powers_vec, pord + 1, pord + 1)
cat("Powers matrix (before solve):\n")
print(powers_matrix)

P <- solve(powers_matrix)
cat("\nP matrix:\n")
print(P)

i1 <- rep(1:(pord + 1), pord + 1) + rep(1:(pord + 1), each = pord + 1)
H <- matrix((1 + (-1)^(i1 - 2))/(i1 - 1), pord + 1, pord + 1)
cat("\nH matrix:\n")
print(H)

W1 <- t(P) %*% H %*% P

cat("\nW1 matrix:\n")
print(W1)
cat("\ndiag(W1):", diag(W1), "\n\n")

# Scale h
h_scaled <- h/2

cat("h scaled by 1/2:", h_scaled[1], "\n\n")

# Build ld
ld0 <- rep(diag(W1), length(h_scaled)) * rep(h_scaled, each = pord + 1)

cat("ld0:\n")
cat("  Length:", length(ld0), "\n")
cat("  First 10:", ld0[1:10], "\n")
cat("  All ld0:", paste(sprintf("%.10f", ld0), collapse=", "), "\n\n")

# Reindex
i1_idx <- c(rep(1:pord, length(h_scaled)) + rep(0:(length(h_scaled) - 1) * (pord + 1), each = pord), length(ld0))

cat("Reindexing:\n")
cat("  i1_idx length:", length(i1_idx), "\n")
cat("  i1_idx (first 10):", i1_idx[1:10], "\n")
cat("  i1_idx (all):", paste(i1_idx, collapse=", "), "\n\n")

ld <- ld0[i1_idx]
ld_after_reindex <- ld  # Save for comparison

cat("ld after reindexing:\n")
cat("  Length:", length(ld), "\n")
cat("  First 10:", ld[1:10], "\n")
cat("  All ld:", paste(sprintf("%.10f", ld), collapse=", "), "\n\n")

# Add overlaps
if (length(h_scaled) > 1) {
    i0 <- 1:(length(h_scaled) - 1) * pord + 1
    i2 <- 1:(length(h_scaled) - 1) * (pord + 1)
    ld[i0] <- ld[i0] + ld0[i2]
}

cat("ld after overlaps:\n")
cat("  First 10:", ld[1:10], "\n")
cat("  All ld:", paste(sprintf("%.10f", ld), collapse=", "), "\n\n")

# Build B matrix
B <- matrix(0, pord + 1, length(ld))
B[1, ] <- ld

for (kk in 1:pord) {
    diwk <- if (kk <= ncol(W1)) diag(W1[, -(1:kk), drop=FALSE]) else numeric(0)
    if (length(diwk) > 0) {
        ind_len <- length(ld) - kk
        pattern <- c(diwk, rep(0, kk - 1))
        values <- (rep(h_scaled, each = pord) * rep(pattern, length(h_scaled)))[1:ind_len]
        B[kk + 1, 1:ind_len] <- values
    }
}

cat("B matrix before bandchol:\n")
cat("  Shape:", dim(B), "\n")
cat("  B[1, 1:10]:", B[1, 1:min(10, ncol(B))], "\n")
if (nrow(B) >= 2) cat("  B[2, 1:10]:", B[2, 1:min(10, ncol(B))], "\n")
if (nrow(B) >= 3) cat("  B[3, 1:10]:", B[3, 1:min(10, ncol(B))], "\n")
cat("  All B[1,]:", paste(sprintf("%.10f", B[1,]), collapse=", "), "\n")
if (nrow(B) >= 2) cat("  All B[2,]:", paste(sprintf("%.10f", B[2, 1:min(20, ncol(B))]), collapse=", "), "...\n")
cat("\n")

# Apply mgcv's bandchol
B_chol <- bandchol(B)

cat("B matrix AFTER bandchol:\n")
cat("  B_chol[1, 1:6]:", B_chol[1, 1:6], "\n")
if (nrow(B_chol) >= 2) cat("  B_chol[2, 1:6]:", B_chol[2, 1:6], "\n")
if (nrow(B_chol) >= 3) cat("  B_chol[3, 1:6]:", B_chol[3, 1:6], "\n")
cat("\n")

# Save B matrices and ld
write.csv(data.frame(ld0=ld0), "/tmp/mgcv_ld0_correct.csv", row.names=FALSE)
write.csv(data.frame(ld_after_reindex=ld_after_reindex), "/tmp/mgcv_ld_after_reindex.csv", row.names=FALSE)
write.csv(data.frame(ld=ld), "/tmp/mgcv_ld_after_overlaps.csv", row.names=FALSE)
write.csv(B, "/tmp/mgcv_B_before_chol.csv", row.names=FALSE)
write.csv(B_chol, "/tmp/mgcv_B_after_chol.csv", row.names=FALSE)

cat("Saved ld and B matrices to /tmp/mgcv_*.csv\n")
cat("Final penalty matrix Frobenius norm:", norm(sm$S[[1]], "F"), "\n")
