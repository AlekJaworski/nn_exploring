#!/usr/bin/env Rscript
# Trace mgcv's gradient values during optimization

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Build design matrix and penalties manually
sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]

cat("\n========== PENALTY MATRIX INFO ==========\n")
cat(sprintf("S1: rank=%d, max=%.6f, frobenius=%.6f\n",
            qr(sm1$S[[1]])$rank,
            max(abs(sm1$S[[1]])),
            sqrt(sum(sm1$S[[1]]^2))))

cat(sprintf("S2: rank=%d, max=%.6f, frobenius=%.6f\n",
            qr(sm2$S[[1]])$rank,
            max(abs(sm2$S[[1]])),
            sqrt(sum(sm2$S[[1]]^2))))

cat("\n========== FITTING WITH TRACE ==========\n")

# Fit with trace to see optimization
ctrl <- gam.control(trace=TRUE, epsilon=1e-7)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\n========== FINAL RESULTS ==========\n")
cat(sprintf("Final sp: [%.6f, %.6f]\n", fit$sp[1], fit$sp[2]))
cat(sprintf("REML: %.6f\n", fit$gcv.ubre))
cat(sprintf("Converged: %s\n", fit$converged))

# Now extract details about the fit
cat("\n========== FIT DETAILS ==========\n")
cat(sprintf("Number of coefficients: %d\n", length(fit$coefficients)))
cat(sprintf("Residual df: %.2f\n", fit$df.residual))
cat(sprintf("EDF: %.2f\n", sum(fit$edf)))

# Check the actual iteration history if available
if (!is.null(fit$outer.info)) {
    cat("\n========== OUTER ITERATION INFO ==========\n")
    print(fit$outer.info)
}
