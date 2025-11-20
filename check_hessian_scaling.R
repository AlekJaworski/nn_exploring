#!/usr/bin/env Rscript
# Get mgcv's gradient AND Hessian at EACH iteration

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

cat("\n========== mgcv ITERATION DETAILS ==========\n")

# Monkey-patch the optimizer to print Hessian at each iteration
# Unfortunately, mgcv doesn't expose this directly, but we can see from trace output

ctrl <- gam.control(trace=TRUE, epsilon=1e-9)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\n========== FINAL HESSIAN (at convergence) ==========\n")
cat("Note: This is at iteration", fit$outer.info$iter, "\n")
print(fit$outer.info$hess)

cat("\n========== COMPARISON ==========\n")
cat("Iteration progression of max|grad|:\n")
cat("  1: 41.61\n")
cat("  2: 29.86\n")
cat("  3: 5.07\n")
cat("  4: 0.24\n")
cat("  5: 0.0006\n")
cat("\nTo get Hessian at iteration 1, we'd need to modify mgcv source.\n")
cat("But we can infer from the fact that gradient decreases monotonically\n")
cat("that the Hessian must be giving correct Newton directions.\n")

cat("\n========== KEY QUESTION ==========\n")
cat("Our Hessian diagonal at iter 1: [2.22e-4, 3.05e-3]\n")
cat("mgcv's final Hessian diagonal: [2.81, 3.19]\n")
cat("Ratio: ~10000x difference!\n\n")
cat("This suggests our Hessian is scaled wrong by a factor of ~10000.\n")
cat("For n=100, total_rank=16:\n")
cat("  (n - total_rank) = 84\n")
cat("  84^2 = 7056 (close to 10000)\n\n")
cat("Hypothesis: We're missing a factor of (n-total_rank) in Hessian!\n")
