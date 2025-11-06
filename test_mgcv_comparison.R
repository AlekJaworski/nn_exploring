#!/usr/bin/env Rscript
# Test what mgcv does with boundary conditions

library(mgcv)

# Train on subset [0.3, 0.7], predict on [0, 1]
set.seed(42)
x_train <- seq(0.3, 0.7, length=50)
y_train <- sin(2 * pi * x_train) + rnorm(50, sd=0.2)

# Fit GAM
gam_fit <- gam(y_train ~ s(x_train, k=10, bs="cr"), method="REML")

# Predict on wider range
x_test <- seq(0, 1, length=50)
# Create a data frame with the correct variable name
test_data <- data.frame(x_train = x_test)
y_pred <- predict(gam_fit, newdata=test_data)

cat("mgcv predictions:\n")
cat(sprintf("  x=0.0 (outside): %.4f\n", y_pred[1]))
cat(sprintf("  x=0.3 (edge):    %.4f\n", y_pred[16]))
cat(sprintf("  x=0.5 (middle):  %.4f\n", y_pred[26]))
cat(sprintf("  x=0.7 (edge):    %.4f\n", y_pred[36]))
cat(sprintf("  x=1.0 (outside): %.4f\n", y_pred[50]))

# Check the basis setup
cat("\nBasis info:\n")
cat(sprintf("  Number of basis functions: %d\n", length(coef(gam_fit))))
cat(sprintf("  Smoothing parameter: %.6f\n", gam_fit$sp))

# Get the smooth object to see knot placement
sm <- gam_fit$smooth[[1]]
cat(sprintf("\nKnot range: [%.3f, %.3f]\n", min(sm$knots), max(sm$knots)))
cat(sprintf("Data range: [%.3f, %.3f]\n", min(x_train), max(x_train)))
