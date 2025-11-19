#!/usr/bin/env Rscript
# Try to extract the actual gradient computation from mgcv

library(mgcv)

# The outer iteration code is in gam.fit3
# Let's look at what it actually does

cat("=== Examining gam.outer function (outer iteration) ===\n\n")

# Try to print the function
gam_outer_body <- body(mgcv::gam.outer)
cat("gam.outer exists:", exists("gam.outer", where="package:mgcv"), "\n\n")

# Look for the key function that computes derivatives
# In Wood (2011), the derivatives are computed in the "performance iteration"
cat("=== Searching for derivative/gradient computation ===\n\n")

# Check if we can see initial.sp
cat("Functions related to smoothing parameter optimization:\n")
all_funcs <- ls("package:mgcv")
sp_funcs <- all_funcs[grepl("sp|smooth|outer", all_funcs, ignore.case=TRUE)]
print(head(sp_funcs, 20))
cat("\n")

# The key is likely in the C code. Let me try to trace what happens
cat("=== Tracing a simple GAM fit ===\n")
set.seed(42)
n <- 50
x <- rnorm(n)
y <- sin(x) + 0.1*rnorm(n)

# Fit with trace
fit <- gam(y ~ s(x, k=8, bs='cr'), method="REML")

cat("\nOuter info structure:\n")
str(fit$outer.info)
cat("\n")

# Look at what outer.info contains
cat("outer.info names:", names(fit$outer.info), "\n")
cat("iter:", fit$outer.info$iter, "\n")
cat("conv:", fit$outer.info$conv, "\n")
if (!is.null(fit$outer.info$grad)) {
  cat("grad:", fit$outer.info$grad, "\n")
}

# Check the smoothing parameter estimation
cat("\nSmoothing parameter info:\n")
cat("sp:", fit$sp, "\n")
cat("method:", fit$method, "\n")

# Look at mgcv source online or in comments
cat("\n=== Key insight from Wood (2011) ===\n")
cat("The REML gradient w.r.t. log(lambda) is computed as:\n")
cat("  grad_i = [tr(F_i) - rank(S_i) + (lambda_i * beta' S_i beta) / phi] / 2\n")
cat("where F_i = A^-1 * lambda_i * S_i\n")
cat("and A = X' W X + sum(lambda_j * S_j)\n")
cat("\nThis is equation (20) in Wood (2011) JRSS-B\n")

