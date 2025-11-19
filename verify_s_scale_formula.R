#!/usr/bin/env Rscript
library(mgcv)

set.seed(42)
n <- 100
X1 <- rnorm(n)
dat <- data.frame(x1=X1)

# Get smooth object BEFORE normalization
sm_raw <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL, scale.penalty=FALSE)[[1]]
cat("=== WITHOUT scale.penalty ===\n")
cat("S.scale:", sm_raw$S.scale, "\n")
if (is.null(sm_raw$S.scale)) cat("S.scale is NULL (not computed)\n")

# Get smooth object WITH normalization (default)
sm_scaled <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL, scale.penalty=TRUE)[[1]]
cat("\n=== WITH scale.penalty ===\n")
cat("S.scale:", sm_scaled$S.scale, "\n")

# Verify the formula manually
X_mat <- sm_raw$X
S_mat <- sm_raw$S[[1]]

cat("\n=== Manual computation ===\n")
norm_X_inf <- max(rowSums(abs(X_mat)))
norm_X_1 <- max(colSums(abs(X_mat)))
norm_X_F <- sqrt(sum(X_mat^2))

cat("||X||_inf (max row sum):", norm_X_inf, "\n")
cat("||X||_1 (max col sum):", norm_X_1, "\n")
cat("||X||_F (Frobenius):", norm_X_F, "\n")
cat("||X||_inf^2:", norm_X_inf^2, "\n")

cat("\nNow using norm() function:\n")
cat("norm(X, 'I'):", norm(X_mat, "I"), "\n")
cat("norm(X, '1'):", norm(X_mat, "1"), "\n")
cat("norm(X, 'F'):", norm(X_mat, "F"), "\n")

norm_S_inf <- max(rowSums(abs(S_mat)))
norm_S_F <- sqrt(sum(S_mat^2))
norm_S_default <- norm(S_mat)  # what does default give?

cat("\n||S||_inf (max row sum):", norm_S_inf, "\n")
cat("||S||_F (Frobenius):", norm_S_F, "\n")  
cat("norm(S) [default]:", norm_S_default, "\n")

# Try the formula
maXX <- norm(X_mat, "I")^2
cat("\nmaXX = ||X||_inf^2 =", maXX, "\n")

# What does norm(S) without type give?
cat("\nTrying formula variants:\n")
cat("||S||_default / maXX =", norm(S_mat) / maXX, "\n")
cat("||S||_inf / maXX =", norm_S_inf / maXX, "\n")
cat("||S||_F / maXX =", norm_S_F / maXX, "\n")

cat("\nActual S.scale from smoothCon:", sm_scaled$S.scale, "\n")
