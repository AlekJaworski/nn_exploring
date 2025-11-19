#!/usr/bin/env Rscript
# Unit test: compute gradient at EXACT sp values and compare

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1

# Save for Python
write.csv(x, "/tmp/unit_x.csv", row.names=FALSE)
write.csv(data.frame(y=y), "/tmp/unit_y.csv", row.names=FALSE)

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Build design matrix
sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]

X <- cbind(sm1$X, sm2$X)
S <- list(sm1$S[[1]], sm2$S[[1]])

cat("\n========== SETUP ==========\n")
cat(sprintf("n=%d, p=%d (X), q1=%d, q2=%d (basis dims)\n",
            n, ncol(X), ncol(sm1$X), ncol(sm2$X)))
cat(sprintf("S1 rank=%d, S2 rank=%d\n",
            qr(S[[1]])$rank, qr(S[[2]])$rank))

# Test at specific sp values
test_sps <- list(
  c(1.0, 1.0),
  c(5.69, 5.20),  # Final converged value from mgcv
  c(0.1, 0.1),
  c(10.0, 1.0)
)

fam <- gaussian()

for (sp in test_sps) {
  cat(sprintf("\n========== sp = [%.4f, %.4f] ==========\n", sp[1], sp[2]))

  # Fit at this sp value
  ctrl <- gam.control(epsilon=1e-7, maxit=1, trace=FALSE)
  fit <- gam.fit3(
    x = X,
    y = y,
    sp = sp,
    Eb = list(),  # No extra blocks
    weights = rep(1, n),
    offset = rep(0, n),
    U1 = diag(ncol(X)),
    Mp = -1,
    family = fam,
    control = ctrl,
    intercept = TRUE,
    deriv = 2,  # Compute gradient and Hessian
    gamma = 1,
    scale = -1,  # Estimate scale
    scoreType = "REML",
    Sl = list(S[[1]], S[[2]])
  )

  cat(sprintf("REML = %.8f\n", fit$REML))
  cat(sprintf("Scale (phi) = %.8f\n", fit$scale.est))
  cat(sprintf("Gradient:\n"))
  print(fit$dgH$dgH[1:length(sp)])
  cat(sprintf("Hessian diagonal:\n"))
  print(diag(fit$dgH$dgH2)[1:length(sp)])
  cat(sprintf("RSS = %.8f\n", sum(fit$residuals^2)))

  # Save results to file for Python to read
  results <- list(
    sp = sp,
    REML = fit$REML,
    scale = fit$scale.est,
    gradient = as.numeric(fit$dgH$dgH[1:length(sp)]),
    hessian_diag = diag(fit$dgH$dgH2)[1:length(sp)],
    RSS = sum(fit$residuals^2),
    edf = sum(fit$edf),
    rank_total = qr(S[[1]])$rank + qr(S[[2]])$rank
  )

  fname <- sprintf("/tmp/mgcv_sp_%.4f_%.4f.json", sp[1], sp[2])
  write(jsonlite::toJSON(results, pretty=TRUE), fname)
  cat(sprintf("Saved to %s\n", fname))
}

cat("\n========== FILES CREATED ==========\n")
cat("/tmp/unit_x.csv\n")
cat("/tmp/unit_y.csv\n")
list.files("/tmp", pattern="mgcv_sp_.*\\.json", full.names=TRUE)
