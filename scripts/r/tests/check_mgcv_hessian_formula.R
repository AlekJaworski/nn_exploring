#!/usr/bin/env Rscript
# Extract the actual Hessian formula from mgcv source
library(mgcv)

# Look for the gdi.c file or R implementation
mgcv_path <- system.file(package = "mgcv")
cat("mgcv package location:", mgcv_path, "\n\n")

# Check for C source files
src_path <- file.path(mgcv_path, "src")
if (dir.exists(src_path)) {
  cat("Source files in", src_path, ":\n")
  print(list.files(src_path, pattern = "\\.(c|h)$"))
  cat("\n")
}

# Look for gdi comments in help files
cat("========== Checking gam.outer documentation ==========\n")
cat("From ?gam.outer:\n")
cat("The outer Newton method computes derivatives of the REML/GCV score\n")
cat("using implicit differentiation.\n\n")

# Print Wood (2011) reference
cat("========== WOOD 2011 FORMULA CHECK ==========\n")
cat("V = -2*l_R (REML criterion to minimize)\n")
cat("\nGradient with respect to ρ = log(λ):\n")
cat("∂V/∂ρ_i = tr(M_i*A) - r_i + β'*M_i*β/φ\n")
cat("\nHessian:\n")
cat("∂²V/∂ρ_i∂ρ_j = -tr(M_i*A*M_j*A) + ∂/∂ρ_j[β'*M_i*β/φ]\n")
cat("\nUsing implicit differentiation:\n")
cat("  ∂β/∂ρ_j = -A*M_j*β\n")
cat("  ∂φ/∂ρ_j = -(2/φ)*β'*M_j*β\n")
cat("\nFull Hessian:\n")
cat("H[i,j] = -tr(M_i*A*M_j*A) + (2*β'*M_i*A*M_j*β)/φ - (2*β'*M_i*β*β'*M_j*β)/φ²\n\n")

cat("========== SIGN CHECK ==========\n")
cat("The Hessian should be POSITIVE DEFINITE for a minimum of V.\n")
cat("mgcv's final Hessian diagonal: [2.813, 3.186] (POSITIVE)\n")
cat("Our Hessian diagonal: [-1.76e-4, 0.0] (NEGATIVE)\n\n")

cat("Possible issues:\n")
cat("1. Sign error in trace term\n")
cat("2. Sign error in penalty derivative terms\n")
cat("3. Missing additional terms\n")
cat("4. Using wrong A matrix (should use current A at each iteration)\n")
