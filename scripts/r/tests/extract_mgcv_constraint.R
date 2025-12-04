#!/usr/bin/env Rscript
library(mgcv)

# Create smooth constructor
dat <- data.frame(x = seq(0, 1, length.out = 100))
sm <- smoothCon(s(x, k=10, bs="cr"), data=dat, knots=NULL)[[1]]

cat("Constraint matrix C:\n")
if (!is.null(sm$C)) {
  print(sm$C)
  cat("  Shape:", dim(sm$C), "\n")
  write.csv(sm$C, "/tmp/mgcv_constraint.csv", row.names=FALSE)
  cat("Saved to /tmp/mgcv_constraint.csv\n")
} else {
  cat("  No constraint matrix\n")
}

cat("\nFull penalty (before constraint):\n")
S_full <- sm$S[[1]]
cat("  Shape:", dim(S_full), "\n")
cat("  Frobenius:", norm(S_full, "F"), "\n")
write.csv(S_full, "/tmp/mgcv_penalty_full.csv", row.names=FALSE)
cat("Saved to /tmp/mgcv_penalty_full.csv\n")
