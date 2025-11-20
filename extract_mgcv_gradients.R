#!/usr/bin/env Rscript
# Extract mgcv's gradient and Hessian values at each iteration

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1

# Save for Python
write.csv(x, "/tmp/unit_x.csv", row.names=FALSE)
write.csv(data.frame(y=y), "/tmp/unit_y.csv", row.names=FALSE)

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

cat("\n========== FITTING WITH DETAILED OUTPUT ==========\n")

# Fit with outer newton method (default)
ctrl <- gam.control(trace=TRUE, epsilon=1e-9)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\n========== OUTER.INFO ==========\n")
print(fit$outer.info)

cat("\n========== CONVERGENCE INFO ==========\n")
cat(sprintf("Converged: %s\n", fit$converged))
cat(sprintf("Iterations: %d\n", fit$outer.info$iter))
cat(sprintf("Final gradient (max|grad|): %.10e\n", max(abs(fit$outer.info$grad))))
cat(sprintf("Final sp: [%.6f, %.6f]\n", fit$sp[1], fit$sp[2]))
cat(sprintf("REML: %.8f\n", fit$gcv.ubre))

# Extract penalty ranks
sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
rank1 <- qr(sm1$S[[1]])$rank
rank2 <- qr(sm2$S[[1]])$rank

cat(sprintf("\nPenalty ranks: [%d, %d], total=%d\n", rank1, rank2, rank1+rank2))
cat(sprintf("n=%d, n-total_rank=%d\n", n, n-(rank1+rank2)))

# Save results
results <- list(
  sp_final = as.numeric(fit$sp),
  REML_final = fit$gcv.ubre,
  iterations = fit$outer.info$iter,
  final_gradient = as.numeric(fit$outer.info$grad),
  final_hessian = as.matrix(fit$outer.info$hess),
  rank1 = rank1,
  rank2 = rank2,
  n = n,
  converged = fit$converged
)

write(jsonlite::toJSON(results, pretty=TRUE), "/tmp/mgcv_final_results.json")
cat("\nSaved to /tmp/mgcv_final_results.json\n")
