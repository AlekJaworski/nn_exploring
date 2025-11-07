#!/usr/bin/env Rscript
# Extract the actual penalty matrix from mgcv

library(mgcv)

# Load data
data <- read.csv("/tmp/test_data.csv")
x <- data$x
y <- data$y

# Fit GAM
gam_fit <- gam(y ~ s(x, k=10, bs="cr"), method="REML")

# Extract penalty matrix
sm <- gam_fit$smooth[[1]]
S <- sm$S[[1]]

cat("Penalty Matrix from mgcv:\n")
cat("Shape:", dim(S), "\n")
cat("Frobenius norm:", norm(S, "F"), "\n")
cat("Trace:", sum(diag(S)), "\n")
cat("Rank:", Matrix::rankMatrix(S)[1], "\n\n")

cat("First 5x5 block:\n")
print(S[1:5, 1:5])

cat("\nDiagonal:\n")
print(diag(S))

# Save to file
write.csv(S, "/tmp/mgcv_penalty_matrix.csv", row.names = FALSE)
cat("\nSaved to /tmp/mgcv_penalty_matrix.csv\n")
