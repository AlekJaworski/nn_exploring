#!/usr/bin/env Rscript

# Extract mgcv's exact h scaling and intermediate values
library(mgcv)

cat(strrep("=", 70), "\n")
cat("EXTRACTING EXACT h VALUES FROM MGCV\n")
cat(strrep("=", 70), "\n\n")

# Setup (matching our test case)
x_min <- 0.0
x_max <- 1.0
k <- 20
degree <- 3
m <- c(2)  # Second derivative
pord <- degree - m[1]  # Should be 1

cat("Parameters:\n")
cat("  k =", k, "\n")
cat("  degree =", degree, "\n")
cat("  m =", m, "\n")
cat("  pord =", pord, "\n\n")

# Create knots (mgcv's way)
nk <- k - degree + 1
x_range <- x_max - x_min
xl <- x_min - x_range * 0.001
xu <- x_max + x_range * 0.001
dx <- (xu - xl) / (nk - 1)
knots_ext <- seq(xl - degree*dx, xu + degree*dx, length.out = nk + 2*degree)

cat("Knot vector:\n")
cat("  Total knots:", length(knots_ext), "\n")
cat("  First 4:", knots_ext[1:4], "\n")
cat("  Last 4:", knots_ext[(length(knots_ext)-3):length(knots_ext)], "\n\n")

# Interior knots
k0 <- knots_ext[(degree+1):(degree+nk)]
cat("Interior knots k0:\n")
cat("  Length:", length(k0), "\n")
cat("  k0[1:4]:", k0[1:4], "\n")
cat("  k0:", paste(sprintf("%.10f", k0), collapse=", "), "\n\n")

# h BEFORE scaling
h_unscaled <- diff(k0)
cat("h BEFORE scaling:\n")
cat("  Length:", length(h_unscaled), "\n")
cat("  h[1:4]:", sprintf("%.10f", h_unscaled[1:4]), "\n")
cat("  All h:", paste(sprintf("%.10f", h_unscaled), collapse=", "), "\n\n")

# h1 for evaluation points (uses UNscaled h!)
h1 <- rep(h_unscaled / pord, each = pord)
cat("h1 (for evaluation points):\n")
cat("  Length:", length(h1), "\n")
cat("  h1[1:6]:", sprintf("%.10f", h1[1:6]), "\n\n")

# k1 evaluation points
k1 <- cumsum(c(k0[1], h1))
cat("k1 (evaluation points):\n")
cat("  Length:", length(k1), "\n")
cat("  k1[1:6]:", sprintf("%.10f", k1[1:6]), "\n")
cat("  k1[(end-5):end]:", sprintf("%.10f", k1[(length(k1)-5):length(k1)]), "\n\n")

# NOW scale h (by 1/2)
h_scaled <- h_unscaled / 2.0
cat("h AFTER scaling (h/2):\n")
cat("  h_scaled[1:4]:", sprintf("%.10f", h_scaled[1:4]), "\n")
cat("  All h_scaled:", paste(sprintf("%.10f", h_scaled), collapse=", "), "\n\n")

# Build W1 matrix
seq_vals <- seq(-1, 1, length.out = pord + 1)
cat("seq_vals:", seq_vals, "\n")

# Build P matrix
powers_matrix <- outer(0:(pord), seq_vals, "^")
P <- solve(t(powers_matrix))
cat("\nP matrix:\n")
print(P)

# Build i1 matrix
i1 <- outer(1:(pord+1), 1:(pord+1), "+")
cat("\ni1 matrix:\n")
print(i1)

# Build H matrix
H <- (1 + (-1)^(i1 - 2)) / (i1 - 1)
cat("\nH matrix:\n")
print(H)

# Build W1
W1 <- t(P) %*% H %*% P
cat("\nW1 matrix:\n")
print(W1)

# Extract diagonal
diag_W1 <- diag(W1)
cat("\ndiag(W1):", sprintf("%.10f", diag_W1), "\n\n")

# Build ld0 (uses SCALED h!)
ld0 <- rep(diag_W1, length(h_scaled)) * rep(h_scaled, each = pord + 1)
cat("ld0 (after scaling):\n")
cat("  Length:", length(ld0), "\n")
cat("  ld0[1:6]:", sprintf("%.10f", ld0[1:6]), "\n")
cat("  All ld0:", paste(sprintf("%.10f", ld0), collapse=", "), "\n\n")

# Reindexing
idx <- c(rep(1:pord, each = length(h_scaled)) + rep((0:(length(h_scaled)-1)) * (pord + 1), pord), length(ld0))
cat("Reindexing:\n")
cat("  idx length:", length(idx), "\n")
cat("  idx[1:6]:", idx[1:6], "\n")
cat("  idx (all):", paste(idx, collapse=", "), "\n\n")

ld <- ld0[idx]
cat("ld (after reindexing):\n")
cat("  Length:", length(ld), "\n")
cat("  ld[1:6]:", sprintf("%.10f", ld[1:6]), "\n")
cat("  All ld:", paste(sprintf("%.10f", ld), collapse=", "), "\n\n")

# Handle overlaps
if (length(h_scaled) > 1) {
    i0 <- (1:(length(h_scaled)-1)) * pord + 1  # R is 1-indexed
    i2 <- (1:(length(h_scaled)-1)) * (pord + 1)
    cat("Overlap adjustment:\n")
    cat("  i0:", i0, "\n")
    cat("  i2:", i2, "\n")
    cat("  ld0[i2]:", sprintf("%.10f", ld0[i2]), "\n")

    ld[i0] <- ld[i0] + ld0[i2]

    cat("  ld after overlaps (at i0):", sprintf("%.10f", ld[i0]), "\n\n")
}

cat("ld (final, after overlaps):\n")
cat("  Length:", length(ld), "\n")
cat("  ld[1:6]:", sprintf("%.10f", ld[1:6]), "\n")
cat("  All ld:", paste(sprintf("%.10f", ld), collapse=", "), "\n\n")

# Save to CSV for comparison
write.csv(data.frame(h_unscaled = c(h_unscaled, rep(NA, max(0, length(ld) - length(h_unscaled))))),
          "/tmp/mgcv_h_unscaled.csv", row.names = FALSE)
write.csv(data.frame(h_scaled = c(h_scaled, rep(NA, max(0, length(ld) - length(h_scaled))))),
          "/tmp/mgcv_h_scaled.csv", row.names = FALSE)
write.csv(data.frame(k1 = k1), "/tmp/mgcv_k1.csv", row.names = FALSE)
write.csv(data.frame(ld0 = ld0), "/tmp/mgcv_ld0.csv", row.names = FALSE)
write.csv(data.frame(ld = ld), "/tmp/mgcv_ld.csv", row.names = FALSE)

cat("\nSaved intermediate values to /tmp/mgcv_*.csv\n")
