#!/usr/bin/env Rscript

library(mgcv)

cat(strrep("=", 70), "\n")
cat("HOW MGCV CALCULATES KNOTS\n")
cat(strrep("=", 70), "\n\n")

# Test with simple data
set.seed(42)
n <- 500
x <- seq(0, 1, length.out = n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.1)

# Get data range
x_min <- min(x)
x_max <- max(x)

cat("Data range: [", x_min, ",", x_max, "]\n\n")

# Test different k values to understand the pattern
for (k in c(10, 20, 30)) {
  cat(strrep("=", 70), "\n")
  cat("k =", k, "\n")
  cat(strrep("=", 70), "\n\n")

  # BS splines
  cat("BS (B-splines):\n")
  gam_bs <- gam(y ~ s(x, k=k, bs="bs"), method="REML")
  sm_bs <- gam_bs$smooth[[1]]

  if (!is.null(sm_bs$knots)) {
    knots_bs <- sm_bs$knots
    cat("  Number of knots:", length(knots_bs), "\n")
    cat("  First knot:", knots_bs[1], "\n")
    cat("  Last knot:", knots_bs[length(knots_bs)], "\n")
    cat("  First knot < x_min?", knots_bs[1] < x_min, "(diff:", knots_bs[1] - x_min, ")\n")
    cat("  Last knot > x_max?", knots_bs[length(knots_bs)] > x_max, "(diff:", knots_bs[length(knots_bs)] - x_max, ")\n")

    # Check spacing
    interior_knots <- knots_bs[knots_bs >= x_min & knots_bs <= x_max]
    cat("  Interior knots:", length(interior_knots), "\n")

    if (length(interior_knots) > 1) {
      spacings <- diff(interior_knots)
      cat("  Interior spacing (first few):", head(spacings, 3), "\n")
      cat("  Spacing uniform?", all(abs(spacings - mean(spacings)) < 1e-10), "\n")
    }
  }

  cat("\n")

  # CR splines
  cat("CR (Cubic Regression):\n")
  gam_cr <- gam(y ~ s(x, k=k, bs="cr"), method="REML")
  sm_cr <- gam_cr$smooth[[1]]

  if (!is.null(sm_cr$xp)) {
    knots_cr <- sm_cr$xp
    cat("  Number of knots:", length(knots_cr), "\n")
    cat("  First knot:", knots_cr[1], "\n")
    cat("  Last knot:", knots_cr[length(knots_cr)], "\n")
    cat("  Spacing (first few):", head(diff(knots_cr), 3), "\n")

    spacings <- diff(knots_cr)
    cat("  Spacing uniform?", all(abs(spacings - mean(spacings)) < 1e-10), "\n")
  }

  cat("\n")
}

cat(strrep("=", 70), "\n")
cat("INVESTIGATING BS KNOT PLACEMENT FORMULA\n")
cat(strrep("=", 70), "\n\n")

# Try to reverse engineer the formula
k <- 20
gam_bs <- gam(y ~ s(x, k=k, bs="bs"), method="REML")
sm_bs <- gam_bs$smooth[[1]]
knots_bs <- sm_bs$knots

cat("k =", k, "\n")
cat("Number of knots:", length(knots_bs), "\n\n")

# Check if it's extending by a fixed fraction of range
x_range <- x_max - x_min
first_extension <- x_min - knots_bs[1]
last_extension <- knots_bs[length(knots_bs)] - x_max

cat("Data range:", x_range, "\n")
cat("Extension before x_min:", first_extension, "\n")
cat("Extension after x_max:", last_extension, "\n")
cat("First extension / range:", first_extension / x_range, "\n")
cat("Last extension / range:", last_extension / x_range, "\n\n")

# Check interior knot spacing
interior_knots <- knots_bs[knots_bs >= x_min & knots_bs <= x_max]
cat("Interior knots:", length(interior_knots), "\n")
interior_spacing <- mean(diff(interior_knots))
cat("Average interior spacing:", interior_spacing, "\n")
cat("Extension / interior_spacing (first):", first_extension / interior_spacing, "\n")
cat("Extension / interior_spacing (last):", last_extension / interior_spacing, "\n\n")

# Look at mgcv source code for smooth.construct.bs.smooth.spec
cat("Checking mgcv smooth.construct.bs.smooth.spec...\n")
cat("Knots placed by: smooth.construct.bs.smooth.spec\n\n")

# Print all knots to see the pattern
cat("All knots:\n")
print(knots_bs)
