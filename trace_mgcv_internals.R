library(mgcv)

# Trace mgcv's internal gradient computation
set.seed(42)
n <- 1000
x <- matrix(rnorm(n*4), n, 4)
y <- sin(x[,1]) + 0.5*x[,2]^2 + cos(x[,3]) + 0.3*x[,4] + rnorm(n)*0.1

# Fit with mgcv to get the setup
cat("=== Fitting with mgcv ===\n")
gam_fit <- gam(y ~ s(x[,1], k=16, bs='cr') + s(x[,2], k=16, bs='cr') +
                   s(x[,3], k=16, bs='cr') + s(x[,4], k=16, bs='cr'),
               method='REML',
               control=gam.control(trace=FALSE, nthreads=1))

cat("\n=== Model Structure ===\n")
cat("Number of coefficients:", length(coef(gam_fit)), "\n")
cat("Number of smooths:", length(gam_fit$smooth), "\n")

# Get the model matrix and penalties
X <- predict(gam_fit, type="lpmatrix")
p <- ncol(X)
cat("Design matrix X dimensions:", dim(X), "\n")

# Get penalty matrices - need to extract them in the full coefficient space
S_list <- list()
cat("\n=== Penalty Matrices ===\n")
# mgcv stores penalties in smooth-specific basis, need to map to full space
for (i in 1:length(gam_fit$smooth)) {
    smooth <- gam_fit$smooth[[i]]
    first_col <- smooth$first.para
    last_col <- smooth$last.para
    n_coef <- last_col - first_col + 1

    cat("Smooth", i, ": columns", first_col, "to", last_col, "(", n_coef, "coefs )\n")

    # Create full penalty matrix
    S_full <- matrix(0, p, p)
    S_smooth <- smooth$S[[1]]
    S_full[first_col:last_col, first_col:last_col] <- S_smooth

    S_list[[i]] <- S_full
    cat("  Penalty rank:", qr(S_smooth)$rank, "\n")
}

# Check mgcv's approach to QR decomposition
cat("\n=== Testing QR approach ===\n")

# Get initial lambda values (at REML optimum)
sp <- gam_fit$sp
cat("Optimal smoothing parameters:", sp, "\n")

# Form the augmented matrix like we do
# X'WX + sum(lambda_i * S_i) then take QR
w <- rep(1, nrow(X))  # weights

# Compute X'WX
XtWX <- t(X) %*% diag(w) %*% X

# Add penalties
A <- XtWX
for (i in 1:length(sp)) {
    A <- A + sp[i] * S_list[[i]]
}

cat("\nA matrix (X'WX + penalties) dimensions:", dim(A), "\n")
cat("A condition number:", kappa(A), "\n")

# Try QR on Cholesky factor
cat("\n=== Cholesky approach ===\n")
R_chol <- tryCatch({
    chol(A)
}, error = function(e) {
    cat("Cholesky failed:", e$message, "\n")
    NULL
})

if (!is.null(R_chol)) {
    cat("Cholesky R diagonal range:", range(diag(R_chol)), "\n")
    P_chol <- solve(R_chol)
    cat("P (from Cholesky) Frobenius norm:", norm(P_chol, "F"), "\n")

    # Compute trace for first penalty
    trace_val <- sum(diag(t(P_chol) %*% S_list[[1]] %*% P_chol))
    cat("Trace (P'*S1*P):", trace_val, "\n")
}

# Try the Z matrix approach that we use
cat("\n=== Z Matrix QR Approach ===\n")

# Form Z matrix
total_rank <- sum(sapply(S_list, function(S) qr(S)$rank))
cat("Total penalty rank:", total_rank, "\n")

# For each penalty, get sqrt - extract only the non-zero block!
penalty_blocks <- list()
for (i in 1:length(sp)) {
    smooth <- gam_fit$smooth[[i]]
    block_start <- smooth$first.para
    block_end <- smooth$last.para

    # Extract just the block for this smooth
    S_block <- smooth$S[[1]]  # This is already the 15×15 block
    lambda <- sp[i]

    # SVD to get square root
    svd_S <- svd(S_block)
    tol <- max(dim(S_block)) * max(svd_S$d) * .Machine$double.eps
    rank <- sum(svd_S$d > tol)

    if (rank > 0) {
        # Sqrt = U * sqrt(D)  (keeping only rank columns)
        # This gives a 15×14 matrix for each penalty
        sqrt_S_block <- svd_S$u[, 1:rank, drop=FALSE] %*% diag(sqrt(svd_S$d[1:rank] * lambda), nrow=rank)
        penalty_blocks[[i]] <- sqrt_S_block
        cat("Penalty", i, ": rank=", rank, ", sqrt_S_block dims=", dim(sqrt_S_block), "\n")
    }
}

# Form Z matrix
sqrt_W_X <- sqrt(w) * X
Z_rows <- nrow(sqrt_W_X) + sum(sapply(penalty_blocks, nrow))
Z <- matrix(0, Z_rows, p)

# Add X'W^{1/2}
Z[1:nrow(sqrt_W_X), ] <- sqrt_W_X

# Add penalty blocks
row_idx <- nrow(sqrt_W_X) + 1
for (i in 1:length(penalty_blocks)) {
    pb <- penalty_blocks[[i]]
    if (!is.null(pb)) {
        # Get column range from smooth info
        smooth <- gam_fit$smooth[[i]]
        block_start <- smooth$first.para
        block_end <- smooth$last.para

        # pb is block_size × rank (e.g., 15×14)
        # We want rank rows × block_size columns in Z
        # So we transpose: t(pb) gives 14×15
        pb_t <- t(pb)
        rank <- nrow(pb_t)

        cat("Penalty", i, "affects columns", block_start, "to", block_end,
            ", adding", rank, "rows\n")

        Z[row_idx:(row_idx + rank - 1), block_start:block_end] <- pb_t
        row_idx <- row_idx + rank
    }
}

cat("\nZ matrix dimensions:", dim(Z), "\n")
cat("Z column norms range:", range(apply(Z, 2, function(col) sqrt(sum(col^2)))), "\n")

# QR decomposition
qr_Z <- qr(Z)
R <- qr.R(qr_Z)
cat("R diagonal range:", range(abs(diag(R))), "\n")

# Compute P = R^{-1}
P_qr <- solve(R)
cat("P (from QR) Frobenius norm:", norm(P_qr, "F"), "\n")

# Compute trace for first penalty
trace_val_qr <- sum(diag(t(P_qr) %*% S_list[[1]] %*% P_qr))
cat("Trace (P'*S1*P) from QR:", trace_val_qr, "\n")

# Compare with Cholesky
if (!is.null(R_chol)) {
    cat("\nFrobenius norm difference (P_chol vs P_qr):", norm(P_chol - P_qr, "F"), "\n")
}
