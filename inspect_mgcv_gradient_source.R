#!/usr/bin/env Rscript
# Find and examine mgcv's gradient computation source

library(mgcv)

# The gradient computation is in gam.fit3
cat("=== Looking for gradient computation code ===\n\n")

# Check what's available
cat("mgcv namespace functions with 'grad' in name:\n")
funcs <- ls("package:mgcv")
grad_funcs <- funcs[grepl("grad", funcs, ignore.case=TRUE)]
print(grad_funcs)
cat("\n")

# The actual gradient is computed in gam.fit3 which calls Sl.initial.repara
# Let's look for the C code that does REML gradient

# First, find where the package is
pkg_path <- find.package("mgcv")
cat("mgcv package location:", pkg_path, "\n")

# Look for source files
src_files <- list.files(file.path(pkg_path, "R"), pattern="\\.R$", full.names=TRUE)
cat("\nR source files:", length(src_files), "\n")

# Search for REML gradient computation
cat("\n=== Searching for REML gradient in R files ===\n")
for (f in src_files) {
  content <- readLines(f, warn=FALSE)
  matches <- grep("REML.*grad|grad.*REML|deriv.*REML", content, ignore.case=TRUE)
  if (length(matches) > 0) {
    cat("\nFile:", basename(f), "\n")
    cat("Matches:\n")
    for (i in matches) {
      cat(sprintf("  Line %d: %s\n", i, trimws(content[i])))
    }
  }
}

# The gradient is likely in C code - look for that
cat("\n=== C/C++ source files ===\n")
c_files <- list.files(file.path(pkg_path, "src"), pattern="\\.(c|cpp|h)$", full.names=TRUE)
if (length(c_files) > 0) {
  cat("Found", length(c_files), "C/C++ files\n")
  for (f in head(c_files, 5)) {
    cat("  ", basename(f), "\n")
  }
}

# Look for .Call references which indicate C functions
cat("\n=== Looking for .Call/.C references in gam.fit3 ===\n")
gam_fit3_code <- capture.output(print(mgcv::gam.fit3))
call_lines <- grep("\\.Call|\\.C\\(", gam_fit3_code)
if (length(call_lines) > 0) {
  cat("Found .Call/.C invocations:\n")
  for (i in call_lines) {
    cat("  ", gam_fit3_code[i], "\n")
  }
}
