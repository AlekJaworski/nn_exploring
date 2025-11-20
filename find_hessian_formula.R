#!/usr/bin/env Rscript
# Search mgcv documentation and code for Hessian formula

library(mgcv)

# Print citation for Wood's paper
cat("\n========== MGCV CITATION ==========\n")
citation("mgcv")

# Check if there's documentation on the REML derivatives
cat("\n========== SEARCHING FOR DERIVATIVE INFO ==========\n")
?gam.outer

# Try to find the actual formula in comments
cat("\n========== Looking for derivative formulas in code ==========\n")
cat("The key functions are implemented in C code (gdi.c)\n")
cat("Formulas should be in Wood (2011) J.R.Statist.Soc.B paper\n")

cat("\n========== WOOD 2011 FORMULA (from paper) ==========\n")
cat("For REML criterion V = -2l_R where l_R is restricted log-likelihood\n")
cat("\nFirst derivative (gradient):\n")
cat("∂V/∂ρ_i = [tr(M_i·A) - r_i + β'·M_i·β/φ]\n")
cat("where:\n")
cat("  ρ_i = log(λ_i)\n")
cat("  M_i = λ_i·S_i\n")
cat("  A = (X'WX + ΣM_i)^(-1)\n")
cat("  r_i = rank(S_i)\n")
cat("  φ = scale parameter\n")

cat("\n\nSecond derivative (Hessian):\n")
cat("∂²V/∂ρ_i∂ρ_j = -tr(M_i·A·M_j·A) + ∂/∂ρ_j[β'·M_i·β/φ]\n")
cat("\nThe second term requires implicit differentiation:\n")
cat("  ∂β/∂ρ_j = -A·M_j·β\n")
cat("  ∂φ/∂ρ_j = -(2/φ)·β'·M_j·β\n")
cat("\nCombining:\n")
cat("  ∂/∂ρ_j[β'·M_i·β/φ] = (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²\n")

cat("\n\nFull Hessian:\n")
cat("H[i,j] = -tr(M_i·A·M_j·A) + (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²\n")

cat("\n========== ACTION ITEMS ==========\n")
cat("1. Verify this formula by checking Wood (2011) paper\n")
cat("2. Implement the missing penalty derivative terms\n")
cat("3. Test that gradient decreases monotonically\n")
