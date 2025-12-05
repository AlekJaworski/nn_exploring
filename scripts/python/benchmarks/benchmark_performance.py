"""
Performance benchmark: Rust vs mgcv for multidimensional GAM gradient computation
Tests both accuracy and speed across various problem sizes
"""
import numpy as np
import pandas as pd
import subprocess
import time
import mgcv_rust

def generate_test_data(n, d, k=10):
    """Generate test data with d smooth terms"""
    np.random.seed(42)
    X_raw = np.random.randn(n, d)

    # True function: sum of smooth functions
    y_true = np.zeros(n)
    for i in range(d):
        y_true += np.sin(2 * np.pi * X_raw[:, i]) + 0.5 * np.cos(4 * np.pi * X_raw[:, i])

    # Add noise
    y = y_true + 0.5 * np.random.randn(n)

    return X_raw, y

def setup_mgcv_matrices(X_raw, y, k=10):
    """Use mgcv to create design matrix and penalty matrices"""
    n, d = X_raw.shape

    # Save data to temp file
    df = pd.DataFrame(X_raw, columns=[f'x{i+1}' for i in range(d)])
    df['y'] = y
    df.to_csv('/tmp/perf_test_data.csv', index=False)

    # Build formula
    smooth_terms = ' + '.join([f's(x{i+1}, k={k}, bs="cr")' for i in range(d)])
    formula = f'y ~ {smooth_terms}'

    # Create GAM in R and extract matrices
    r_code = f"""
    library(mgcv)
    df <- read.csv('/tmp/perf_test_data.csv')

    # Fit GAM to get design matrix and penalties
    fit <- gam({formula}, data=df, method='REML')

    # Extract design matrix
    X <- predict(fit, type='lpmatrix')

    # Extract penalty matrices and expand to full size
    p <- ncol(X)
    n_smooth <- {d}

    penalties <- list()
    for (i in 1:n_smooth) {{
        S_small <- fit$smooth[[i]]$S[[1]]
        start <- fit$smooth[[i]]$first.para
        end <- fit$smooth[[i]]$last.para

        S_full <- matrix(0, p, p)
        S_full[start:end, start:end] <- S_small
        penalties[[i]] <- S_full
    }}

    # Save matrices
    write.csv(X, '/tmp/perf_X.csv', row.names=FALSE)
    for (i in 1:n_smooth) {{
        write.csv(penalties[[i]], paste0('/tmp/perf_S', i, '.csv'), row.names=FALSE)
    }}

    cat('Setup complete\\n')
    cat('Design matrix:', nrow(X), 'x', ncol(X), '\\n')
    cat('Number of smooths:', n_smooth, '\\n')
    """

    result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)
    if result.returncode != 0:
        print("R Error:", result.stderr)
        raise RuntimeError("Failed to setup mgcv matrices")

    # Load matrices
    X = pd.read_csv('/tmp/perf_X.csv').values
    penalties = []
    for i in range(d):
        S = pd.read_csv(f'/tmp/perf_S{i+1}.csv').values
        penalties.append(S)

    return X, penalties

def benchmark_mgcv_gradient(X, penalties, y, lambdas, n_iter=10):
    """Benchmark mgcv gradient computation"""
    n, p = X.shape
    n_smooth = len(penalties)

    # Save matrices for R
    pd.DataFrame(X).to_csv('/tmp/bench_X.csv', index=False)
    pd.DataFrame({'y': y}).to_csv('/tmp/bench_y.csv', index=False)
    for i, S in enumerate(penalties):
        pd.DataFrame(S).to_csv(f'/tmp/bench_S{i+1}.csv', index=False)

    lambda_str = ', '.join([str(l) for l in lambdas])

    r_code = f"""
    library(mgcv)

    X <- as.matrix(read.csv('/tmp/bench_X.csv'))
    y <- read.csv('/tmp/bench_y.csv')$y
    n <- nrow(X)
    p <- ncol(X)

    # Load penalties
    penalties <- list()
    for (i in 1:{n_smooth}) {{
        penalties[[i]] <- as.matrix(read.csv(paste0('/tmp/bench_S', i, '.csv')))
    }}

    lambda <- c({lambda_str})

    # Gradient computation function
    compute_gradient <- function() {{
        # Build A matrix
        XtX <- t(X) %*% X
        A <- XtX
        for (i in 1:{n_smooth}) {{
            A <- A + lambda[i] * penalties[[i]]
        }}

        # Solve for beta
        Ainv <- solve(A)
        beta <- Ainv %*% (t(X) %*% y)

        # Residuals
        fitted <- X %*% beta
        residuals <- y - fitted
        rss <- sum(residuals^2)

        # edf and phi
        edf_total <- sum(diag(Ainv %*% XtX))
        phi <- rss / (n - edf_total)

        # Compute gradient for each penalty (with IFT)
        gradient <- numeric({n_smooth})
        for (i in 1:{n_smooth}) {{
            # Component 1: trace
            trace <- sum(diag(Ainv %*% (lambda[i] * penalties[[i]])))

            # Component 2: implicit beta derivative
            lambda_s_beta <- lambda[i] * (penalties[[i]] %*% beta)
            dbeta_drho <- -Ainv %*% lambda_s_beta

            # drss/drho
            x_dbeta <- X %*% dbeta_drho
            drss_drho <- -2 * sum(residuals * x_dbeta)

            # dedf/drho
            ainv_xtx_ainv <- Ainv %*% XtX %*% Ainv
            dedf_drho <- -sum(diag(ainv_xtx_ainv %*% (lambda[i] * penalties[[i]])))

            # Total
            gradient[i] <- trace + dedf_drho * (-log(phi) + 1) + drss_drho / phi
        }}

        return(gradient)
    }}

    # Warmup
    for (i in 1:5) {{
        g <- compute_gradient()
    }}

    # Benchmark
    times <- numeric({n_iter})
    for (i in 1:{n_iter}) {{
        start <- Sys.time()
        gradient <- compute_gradient()
        end <- Sys.time()
        times[i] <- as.numeric(end - start, units='secs')
    }}

    cat('MGCV_RESULT\\n')
    cat('gradient:', paste(sprintf('%.10f', gradient), collapse=','), '\\n')
    cat('mean_time:', mean(times), '\\n')
    cat('std_time:', sd(times), '\\n')
    """

    result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)

    # Parse results
    lines = result.stdout.split('\n')
    gradient = None
    mean_time = None
    std_time = None

    for line in lines:
        if line.startswith('gradient:'):
            gradient = np.array([float(x) for x in line.split(':')[1].strip().split(',')])
        elif line.startswith('mean_time:'):
            mean_time = float(line.split(':')[1].strip())
        elif line.startswith('std_time:'):
            std_time = float(line.split(':')[1].strip())

    return gradient, mean_time, std_time

def benchmark_rust_gradient(X, penalties, y, lambdas, n_iter=10):
    """Benchmark Rust gradient computation"""
    n = len(y)
    w = np.ones(n)

    # Warmup
    for _ in range(5):
        mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, penalties)

    # Benchmark
    times = []
    for _ in range(n_iter):
        start = time.time()
        gradient = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, penalties)
        end = time.time()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return gradient, mean_time, std_time

def run_benchmark(n, d, k=10):
    """Run complete benchmark for given problem size"""
    print(f"\n{'='*80}")
    print(f"Benchmark: n={n}, d={d} smooths, k={k} basis functions each")
    print(f"{'='*80}\n")

    # Generate data
    print(f"Generating test data...")
    X_raw, y = generate_test_data(n, d, k)

    # Setup matrices via mgcv
    print(f"Setting up design matrix and penalties via mgcv...")
    X, penalties = setup_mgcv_matrices(X_raw, y, k)
    p = X.shape[1]

    print(f"  Design matrix: {n}×{p}")
    print(f"  Penalties: {d} matrices of size {p}×{p}")

    # Random smoothing parameters
    np.random.seed(123)
    lambdas = np.random.uniform(0.1, 10.0, d)
    print(f"  Smoothing parameters: λ = {lambdas}")
    print()

    # Benchmark mgcv (R)
    print("Benchmarking mgcv (R)...")
    grad_mgcv, time_mgcv, std_mgcv = benchmark_mgcv_gradient(X, penalties, y, lambdas, n_iter=10)
    print(f"  Mean time: {time_mgcv*1000:.2f} ± {std_mgcv*1000:.2f} ms")
    print(f"  Gradient: {grad_mgcv}")

    # Benchmark Rust
    print("\nBenchmarking Rust...")
    grad_rust, time_rust, std_rust = benchmark_rust_gradient(X, penalties, y, lambdas, n_iter=10)
    print(f"  Mean time: {time_rust*1000:.2f} ± {std_rust*1000:.2f} ms")
    print(f"  Gradient: {grad_rust}")

    # Compare
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    # Accuracy
    abs_error = np.abs(grad_rust - grad_mgcv)
    rel_error = abs_error / (np.abs(grad_mgcv) + 1e-10)
    max_abs_error = np.max(abs_error)
    max_rel_error = np.max(rel_error)

    print(f"\nAccuracy:")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Max relative error: {max_rel_error:.2%}")
    if max_abs_error < 1e-4:
        print(f"  ✅ EXCELLENT - Matches to numerical precision")
    elif max_rel_error < 0.01:
        print(f"  ✅ GOOD - < 1% error")
    else:
        print(f"  ⚠️  WARNING - Error > 1%")

    # Speed
    speedup = time_mgcv / time_rust
    print(f"\nSpeed:")
    print(f"  mgcv (R):  {time_mgcv*1000:.2f} ± {std_mgcv*1000:.2f} ms")
    print(f"  Rust:      {time_rust*1000:.2f} ± {std_rust*1000:.2f} ms")
    print(f"  Speedup:   {speedup:.2f}x")

    if speedup > 1:
        print(f"  ✅ Rust is {speedup:.2f}x FASTER")
    else:
        print(f"  ⚠️  Rust is {1/speedup:.2f}x SLOWER")

    return {
        'n': n,
        'd': d,
        'p': p,
        'max_abs_error': max_abs_error,
        'max_rel_error': max_rel_error,
        'time_mgcv': time_mgcv,
        'time_rust': time_rust,
        'speedup': speedup
    }

if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK: Rust vs mgcv")
    print("="*80)

    # Test cases: (n, d)
    test_cases = [
        (100, 2),   # Small: 100 obs, 2 smooths
        (500, 3),   # Medium: 500 obs, 3 smooths
        (1000, 5),  # Large: 1000 obs, 5 smooths
    ]

    results = []
    for n, d in test_cases:
        result = run_benchmark(n, d, k=10)
        results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'Problem':<20} {'p':<8} {'Accuracy':<15} {'mgcv (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
    print("-"*80)

    for r in results:
        problem = f"n={r['n']}, d={r['d']}"
        accuracy = f"{r['max_rel_error']:.2e}"
        mgcv_time = f"{r['time_mgcv']*1000:.2f}"
        rust_time = f"{r['time_rust']*1000:.2f}"
        speedup = f"{r['speedup']:.2f}x"

        print(f"{problem:<20} {r['p']:<8} {accuracy:<15} {mgcv_time:<15} {rust_time:<15} {speedup:<10}")

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)

    avg_speedup = np.mean([r['speedup'] for r in results])
    max_error = np.max([r['max_rel_error'] for r in results])

    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Maximum error: {max_error:.2e}")

    if max_error < 1e-4 and avg_speedup > 1:
        print("\n✅ SUCCESS: Rust implementation is both ACCURATE and FAST!")
    elif max_error < 1e-4:
        print("\n✅ Rust implementation is ACCURATE (matches mgcv to numerical precision)")
    else:
        print("\n⚠️  WARNING: Some accuracy issues detected")
