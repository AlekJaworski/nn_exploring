#!/usr/bin/env Rscript
# Compare gradient computation at a fixed point

library(mgcv)

set.seed(42)
n <- 100
X1 <- rnorm(n)
X2 <- rnorm(n)
y <- sin(X1) + 0.5*X2^2 + 0.1*rnorm(n)

# Fit GAM to get structure
dat <- data.frame(x1=X1, x2=X2, y=y)
fit <- gam(y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr'), 
           data=dat, method="REML")

cat("=== Fixed Point Data ===\n")
cat("Optimized sp:", fit$sp, "\n")
cat("S.scale values:", sapply(fit$smooth, function(s) s$S.scale), "\n")
cat("Effective lambda:", fit$sp * sapply(fit$smooth, function(s) s$S.scale), "\n")
cat("\n")

# Get smoothCon objects to access raw data
sm1 <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL)[[1]]
sm2 <- smoothCon(s(x2, k=8, bs='cr'), data=dat, knots=NULL)[[1]]

cat("=== Penalty Matrix Details ===\n")
S1 <- sm1$S[[1]]
S2 <- sm2$S[[1]]
cat("S1 norm:", max(rowSums(abs(S1))), "\n")
cat("S2 norm:", max(rowSums(abs(S2))), "\n")
cat("S1 trace:", sum(diag(S1)), "\n")
cat("S2 trace:", sum(diag(S2)), "\n")
cat("\n")

# Save data for Rust comparison
cat("=== Data for Rust ===\n")
cat("n =", n, "\n")
cat("X1 (first 10):", head(X1, 10), "\n")
cat("X2 (first 10):", head(X2, 10), "\n")
cat("y (first 10):", head(y, 10), "\n")
cat("\n")

# Try to access gradient computation internals
cat("=== Gradient at Solution ===\n")
cat("Gradient:", fit$outer.info$grad, "\n")
cat("Iterations:", fit$outer.info$iter, "\n")
cat("\n")

# Save full data to file for Rust to load
write.table(data.frame(x1=X1, x2=X2, y=y), 
            "fixed_point_data.csv", 
            row.names=FALSE, sep=",")

cat("Data saved to fixed_point_data.csv\n")
