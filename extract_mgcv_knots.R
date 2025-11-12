#!/usr/bin/env Rscript
library(mgcv)

# Load data
data <- read.csv("/tmp/test_data.csv")
x <- data$x
y <- data$y

# Fit GAM
gam_fit <- gam(y ~ s(x, k=10, bs="cr"), method="REML")

# Extract smooth object
sm <- gam_fit$smooth[[1]]

cat("Smooth object details:\n")
cat("  bs.dim (number of basis):", sm$bs.dim, "\n")
cat("  df:", sm$df, "\n")

# Extract knots
cat("\nKnots used by mgcv:\n")
print(sm$xp)
cat("  Number of knots:", length(sm$xp), "\n")

# Save knots
write.csv(data.frame(knots=sm$xp), "/tmp/mgcv_knots.csv", row.names=FALSE)
cat("\nSaved to /tmp/mgcv_knots.csv\n")
