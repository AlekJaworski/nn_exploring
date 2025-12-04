#!/usr/bin/env Rscript
# Compute mgcv's gradient at EXACTLY our starting lambda values

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
write.csv(x, "/tmp/unit_x.csv", row.names=FALSE)
write.csv(data.frame(y=y), "/tmp/unit_y.csv", row.names=FALSE)

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Build design matrix
sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
X <- cbind(sm1$X, sm2$X)
S <- list(sm1$S[[1]], sm2$S[[1]])

cat("========== OUR STARTING LAMBDA ==========\n")
our_lambda <- c(0.0030637181, 0.0030431402)
cat("lambda:", our_lambda, "\n\n")

cat("========== MGCV GRADIENT AT OUR LAMBDA ==========\n")
fam <- gaussian()
ctrl <- gam.control(epsilon=1e-7, maxit=1, trace=FALSE)

fit <- gam.fit3(
  x = X,
  y = y,
  sp = our_lambda,
  Eb = list(),
  weights = rep(1, n),
  offset = rep(0, n),
  U1 = diag(ncol(X)),
  Mp = -1,
  family = fam,
  control = ctrl,
  intercept = TRUE,
  deriv = 2,  # Compute gradient and Hessian
  gamma = 1,
  scale = -1,
  scoreType = "REML",
  Sl = S
)

cat("REML:", fit$REML, "\n")
cat("Gradient:", fit$dgH$dgH[1:2], "\n")
cat("Gradient magnitude:", abs(fit$dgH$dgH[1:2]), "\n")
cat("Max |gradient|:", max(abs(fit$dgH$dgH[1:2])), "\n\n")

cat("========== COMPARISON ==========\n")
cat("mgcv gradient at λ=0.003:", max(abs(fit$dgH$dgH[1:2])), "\n")
cat("Our gradient at λ=0.003: 3.99\n")
cat("Ratio:", max(abs(fit$dgH$dgH[1:2])) / 3.99, "\n")
