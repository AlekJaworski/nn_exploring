# Generate test data for Rust agreement test

set.seed(123)
n <- 500
d <- 1  # 1D for simplicity
k <- 20  # basis size

x <- runif(n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)

# Write in format expected by test_agreement.rs
cat(n, d, k, "\n", file="test_data.txt")

# Write x values (one row per observation)
for (i in 1:n) {
  cat(x[i], "\n", file="test_data.txt", append=TRUE)
}

# Write y values (one per line)
for (i in 1:n) {
  cat(y[i], "\n", file="test_data.txt", append=TRUE)
}

cat("Test data written to test_data.txt\n")
cat("Format: n=", n, ", d=", d, ", k=", k, "\n")
