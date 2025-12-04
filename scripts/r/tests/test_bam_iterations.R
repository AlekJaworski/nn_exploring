# Test bam() convergence and iteration count

library(mgcv)

set.seed(123)
n <- 500
x <- runif(n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)

cat("=== Testing bam() convergence ===\n\n")

# Test regular gam with REML (default Newton)
cat("1. gam() with method='REML' (Newton by default):\n")
m_gam <- gam(y ~ s(x, k=20, bs="cr"), method="REML")
cat("   Lambda:", m_gam$sp, "\n")
cat("   Outer iterations:", m_gam$outer.info$iter, "\n")
cat("   Convergence:", m_gam$outer.info$conv, "\n\n")

# Test gam with Fellner-Schall
cat("2. gam() with optimizer='efs' (Fellner-Schall):\n")
m_fs <- gam(y ~ s(x, k=20, bs="cr"), method="REML", optimizer="efs")
cat("   Lambda:", m_fs$sp, "\n")
cat("   Outer iterations:", m_fs$outer.info$iter, "\n")
cat("   Convergence:", m_fs$outer.info$conv, "\n\n")

# Test bam (uses different PIRLS approach)
cat("3. bam() with method='REML':\n")
m_bam <- bam(y ~ s(x, k=20, bs="cr"), method="REML")
cat("   Lambda:", m_bam$sp, "\n")
cat("   Outer iterations:", m_bam$outer.info$iter, "\n")
cat("   Convergence:", m_bam$outer.info$conv, "\n\n")

# Test bam with fREML (fast REML)
cat("4. bam() with method='fREML':\n")
m_bam_freml <- bam(y ~ s(x, k=20, bs="cr"), method="fREML")
cat("   Lambda:", m_bam_freml$sp, "\n")
cat("   Outer iterations:", m_bam_freml$outer.info$iter, "\n")
cat("   Convergence:", m_bam_freml$outer.info$conv, "\n\n")

cat("=== Summary ===\n\n")
cat("Method               Lambda      Iterations\n")
cat("------               ------      ----------\n")
cat(sprintf("gam(REML)           %.4f      %d\n", m_gam$sp, m_gam$outer.info$iter))
cat(sprintf("gam(efs)            %.4f      %d\n", m_fs$sp, m_fs$outer.info$iter))
cat(sprintf("bam(REML)           %.4f      %d\n", m_bam$sp, m_bam$outer.info$iter))
cat(sprintf("bam(fREML)          %.4f      %d\n", m_bam_freml$sp, m_bam_freml$outer.info$iter))
