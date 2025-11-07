#!/usr/bin/env Rscript
library(mgcv)

# Create smooth constructor for CR splines
dat <- data.frame(x = seq(0, 1, length.out = 100))
sm <- smoothCon(s(x, k=10, bs="cr"), data=dat, knots=NULL)[[1]]

cat("CR Spline Properties:\n")
cat("  Class:", class(sm), "\n")
cat("  Number of basis functions:", sm$bs.dim, "\n")
cat("  Boundary knots:", sm$Boundary.knots, "\n")
cat("  Degree:", sm$p.order - 1, "\n")

# Check the penalty matrix
S <- sm$S[[1]]
cat("\nPenalty Matrix:\n")
cat("  Shape:", dim(S), "\n")
cat("  Type: ", attr(sm$S[[1]], "type"), "\n")
cat("  Frobenius:", norm(S, "F"), "\n")

# Check if there's any documentation
cat("\nSmooth spec:\n")
print(sm$label)
print(sm$term)

# Look for penalty construction method
cat("\nLooking for penalty construction info...\n")
if (!is.null(sm$UZ)) {
  cat("  Has UZ matrix:", dim(sm$UZ), "\n")
}
if (!is.null(sm$C)) {
  cat("  Has constraint matrix C:", dim(sm$C), "\n")
}
