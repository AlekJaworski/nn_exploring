# MGCV-RS: Generalized Additive Models in Rust

A Rust implementation of Generalized Additive Models (GAMs) inspired by the R package `mgcv` by Simon Wood.

## Features

### Smooth Basis Functions

- **Cubic Regression Splines**: Natural cubic splines with automatic knot placement
- **Thin Plate Regression Splines**: Optimal smoothing for multidimensional data
- **P-Splines**: Penalized B-splines with difference penalties

### Automatic Smoothing Parameter Selection

- **REML (Restricted Maximum Likelihood)**: The gold standard for smoothing parameter selection
  - Optimizes the marginal likelihood of the data
  - Accounts for uncertainty in coefficient estimates
  - Provides automatic selection of λ (smoothing parameter)

- **GCV (Generalized Cross-Validation)**: Alternative criterion
  - Computationally efficient
  - Asymptotically equivalent to REML
  - Supports gamma parameter for extra penalty on degrees of freedom

### Distribution Families

- **Gaussian** (identity link)
- **Binomial** (logit link)
- **Poisson** (log link)

### Fitting Algorithm

- **PIRLS (Penalized Iteratively Reweighted Least Squares)**:
  - Handles non-Gaussian responses
  - Iteratively solves weighted penalized regression
  - Converges to maximum penalized likelihood estimates

## Architecture

### Module Structure

```
mgcv-rs/
├── src/
│   ├── lib.rs           # Main library interface
│   ├── errors.rs        # Error types
│   ├── linalg.rs        # Basic linear algebra (custom implementation)
│   ├── basis.rs         # Smooth basis functions
│   ├── family.rs        # Exponential family distributions
│   ├── smooth.rs        # Penalized regression
│   ├── reml.rs          # REML smoothing parameter selection
│   ├── gcv.rs           # GCV smoothing parameter selection
│   ├── gam.rs           # Main GAM fitting engine
│   └── main.rs          # Example application
├── tests/
│   └── integration_tests.rs
└── Cargo.toml
```

### Key Concepts

#### 1. Penalized Regression

GAMs are fit by minimizing:

```
∑ᵢ (yᵢ - f(xᵢ))² + λ ∫ [f''(x)]² dx
```

where:
- The first term measures fit to data
- The second term penalizes roughness
- λ controls the trade-off (smoothing parameter)

This is solved as:

```
β̂ = (X^T W X + λS)^{-1} X^T W y
```

where:
- X is the basis matrix
- S is the penalty matrix
- W is the weight matrix (for non-Gaussian families)

#### 2. REML Smoothing Parameter Selection

REML optimizes λ by maximizing:

```
L_REML(λ) = -1/2 [log|X^T X + λS| - log|X^T X| + (n-p)log(σ²) + RSS/σ²]
```

This provides automatic, data-driven selection of the smoothing parameter.

#### 3. Basis Functions

**Cubic Splines**: Piecewise cubic polynomials with continuous second derivatives

**Thin Plate Splines**: Minimize bending energy:

```
∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dx dy
```

**P-Splines**: B-spline basis with difference penalty on coefficients

#### 4. PIRLS Algorithm

For non-Gaussian families:

1. Initialize μ from y
2. Compute η = g(μ) (link function)
3. Repeat until convergence:
   - Compute working weights: w = (∂μ/∂η)² / V(μ)
   - Compute working response: z = η + (y - μ)/(∂μ/∂η)
   - Solve weighted penalized least squares:
     β = (X^T W X + λS)^{-1} X^T W z
   - Update η = Xβ, μ = g^{-1}(η)

## Dependencies

```toml
[dependencies]
ndarray = "0.15"
ndarray-linalg = "0.16"
openblas-src = { version = "0.10", features = ["cblas", "system"] }
rand = "0.8"
```

## Usage

### Basic Example: Fitting a Sine Wave

```rust
use mgcv::prelude::*;
use ndarray::Array1;

fn main() -> Result<()> {
    // Generate data
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 100);
    let y = x.mapv(|xi| xi.sin() + 0.1 * rand::random::<f64>());

    // Create GAM with REML smoothing parameter selection
    let mut gam = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::REML,
    );

    // Add a cubic spline smooth
    let basis = CubicSpline::new(&x.view(), 15)?;
    gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);

    // Fit the model (REML will optimize lambda automatically)
    gam.fit(&[x.view()], &y.view(), 50, 1e-6)?;

    // Print summary
    if let Some(summary) = gam.summary() {
        println!("EDF: {:.2}", summary.edf);
        println!("Deviance: {:.4}", summary.deviance);
        println!("Lambda: {:.6}", summary.lambdas[0]);
    }

    // Predict
    let x_pred = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 200);
    let y_pred = gam.predict(&[x_pred.view()])?;

    Ok(())
}
```

### Poisson GAM Example

```rust
use mgcv::prelude::*;
use ndarray::Array1;

// Count data
let x = Array1::linspace(0.0, 10.0, 100);
let lambda = x.mapv(|xi| (xi / 2.0).exp());
let y = lambda.mapv(|l| /* sample from Poisson(l) */);

let mut gam = GAM::new(
    Box::new(Poisson),
    SmoothingMethod::REML,
);

let basis = PSpline::new(&x.view(), 20, 3)?;
gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);

gam.fit(&[x.view()], &y.view(), 50, 1e-6)?;
```

### Binomial GAM Example

```rust
use mgcv::prelude::*;
use ndarray::Array1;

// Binary response data
let x = Array1::linspace(-3.0, 3.0, 100);
let p = x.mapv(|xi| 1.0 / (1.0 + (-xi).exp()));
let y = p.mapv(|pi| if rand::random::<f64>() < pi { 1.0 } else { 0.0 });

let mut gam = GAM::new(
    Box::new(Binomial),
    SmoothingMethod::REML,
);

let basis = CubicSpline::new(&x.view(), 10)?;
gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);

gam.fit(&[x.view()], &y.view(), 50, 1e-6)?;
```

## Comparison with R's mgcv

### Similarities

- REML-based smoothing parameter selection
- Multiple smooth basis types
- PIRLS fitting algorithm
- Support for multiple GLM families

### Differences

- Rust provides:
  - Type safety
  - Memory safety without garbage collection
  - Better performance for large datasets
  - Easy parallelization

- R's mgcv provides:
  - More mature implementation
  - Extensive diagnostics and plotting
  - Wider range of smooth types
  - Better documentation

## Implementation Details

### REML Optimization

The REML score is minimized using golden section search on log(λ):

```rust
pub fn optimize(
    X: &Array2<f64>,
    S: &Array2<f64>,
    y: &ArrayView1<f64>,
    lambda_min: f64,
    lambda_max: f64,
    tol: f64,
) -> Result<f64>
```

For multiple smoothing parameters, we use gradient descent with finite differences.

### Numerical Stability

- All optimization is done on log(λ) scale
- Cholesky decomposition for solving linear systems
- Determinants computed via Cholesky: det(A) = [∏ᵢ L_ii]²
- Careful handling of edge cases in family functions

### Effective Degrees of Freedom

The effective degrees of freedom (EDF) is computed as:

```
EDF = tr(F)
```

where F = (X^T X)(X^T X + λS)^{-1}

This represents the "equivalent" number of parameters used by the model.

## Testing

Run the test suite:

```bash
cargo test
```

Run a specific test:

```bash
cargo test test_gam_with_reml
```

Run with output:

```bash
cargo test -- --nocapture
```

## Performance Considerations

- Matrix operations are O(n³) in the number of basis functions
- REML optimization requires O(k) REML evaluations where k ~ 20-100
- For large datasets (n > 10,000), consider:
  - Reducing number of basis functions
  - Using P-splines (sparse penalty matrices)
  - Parallelizing smooth fitting

## Future Enhancements

- [ ] Tensor product smooths for interactions
- [ ] Adaptive knot selection
- [ ] Parallel optimization of multiple λ
- [ ] Standard errors and confidence intervals
- [ ] AIC/BIC model selection
- [ ] Residual diagnostics
- [ ] More family types (Gamma, negative binomial, etc.)
- [ ] Mixed model smooths (random effects)
- [ ] Spatial smoothing
- [ ] Plotting utilities

## References

1. Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.

2. Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36.

3. Wood, S.N. (2004). Stable and efficient multiple smoothing parameter estimation for generalized additive models. *Journal of the American Statistical Association*, 99(467), 673-686.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Authors

Built with inspiration from Simon Wood's mgcv package and modern Rust best practices.
