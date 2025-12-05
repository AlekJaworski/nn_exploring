#!/usr/bin/env Rscript

library(mgcv)

cat("MGCV SOURCE CODE FOR KNOT PLACEMENT\n")
cat(strrep("=", 70), "\n\n")

# Get the smooth.construct.bs.smooth.spec function
cat("BS spline knot construction:\n")
cat(strrep("-", 70), "\n")

# Print the function
bs_func <- smooth.construct.bs.smooth.spec

# Get source if possible
tryCatch({
  source_file <- getSrcFilename(bs_func)
  cat("Source file:", source_file, "\n\n")
}, error = function(e) {
  cat("Source file not available\n\n")
})

# Print the function body (relevant parts)
cat("Function body:\n")
print(body(bs_func))

cat("\n\n")
cat(strrep("=", 70), "\n")
cat("CR spline knot construction:\n")
cat(strrep("-", 70), "\n")

cr_func <- smooth.construct.cr.smooth.spec
print(body(cr_func))
