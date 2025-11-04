use ndarray::{Array1, ArrayView1};

/// Exponential family distribution for GAMs
pub trait Family {
    /// Link function: eta = g(mu)
    fn link(&self, mu: &ArrayView1<f64>) -> Array1<f64>;

    /// Inverse link function: mu = g^{-1}(eta)
    fn linkinv(&self, eta: &ArrayView1<f64>) -> Array1<f64>;

    /// Derivative of inverse link: d(mu)/d(eta)
    fn mu_eta(&self, eta: &ArrayView1<f64>) -> Array1<f64>;

    /// Variance function: V(mu)
    fn variance(&self, mu: &ArrayView1<f64>) -> Array1<f64>;

    /// Deviance residuals
    fn dev_resids(&self, y: &ArrayView1<f64>, mu: &ArrayView1<f64>, wt: &ArrayView1<f64>) -> Array1<f64>;

    /// Initialize mu values
    fn initialize(&self, y: &ArrayView1<f64>) -> Array1<f64>;

    /// Validate mu values
    fn validmu(&self, mu: &ArrayView1<f64>) -> bool;

    /// Validate eta values
    fn valideta(&self, eta: &ArrayView1<f64>) -> bool;
}

/// Gaussian family with identity link
pub struct Gaussian;

impl Family for Gaussian {
    fn link(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        mu.to_owned()
    }

    fn linkinv(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        eta.to_owned()
    }

    fn mu_eta(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        Array1::ones(eta.len())
    }

    fn variance(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())
    }

    fn dev_resids(&self, y: &ArrayView1<f64>, mu: &ArrayView1<f64>, wt: &ArrayView1<f64>) -> Array1<f64> {
        (y - mu).mapv(|x| x * x) * wt
    }

    fn initialize(&self, y: &ArrayView1<f64>) -> Array1<f64> {
        y.to_owned()
    }

    fn validmu(&self, _mu: &ArrayView1<f64>) -> bool {
        true
    }

    fn valideta(&self, _eta: &ArrayView1<f64>) -> bool {
        true
    }
}

/// Binomial family with logit link
pub struct Binomial;

impl Family for Binomial {
    fn link(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        mu.mapv(|m| (m / (1.0 - m)).ln())
    }

    fn linkinv(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        eta.mapv(|e| {
            if e > 30.0 {
                1.0
            } else if e < -30.0 {
                0.0
            } else {
                1.0 / (1.0 + (-e).exp())
            }
        })
    }

    fn mu_eta(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        eta.mapv(|e| {
            let exp_e = e.exp();
            exp_e / (1.0 + exp_e).powi(2)
        })
    }

    fn variance(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        mu.mapv(|m| m * (1.0 - m))
    }

    fn dev_resids(&self, y: &ArrayView1<f64>, mu: &ArrayView1<f64>, wt: &ArrayView1<f64>) -> Array1<f64> {
        let mut dev = Array1::zeros(y.len());
        for i in 0..y.len() {
            let yi = y[i];
            let mui = mu[i].max(1e-10).min(1.0 - 1e-10);

            let d1 = if yi > 0.0 { yi * (yi / mui).ln() } else { 0.0 };
            let d2 = if yi < 1.0 { (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln() } else { 0.0 };
            dev[i] = 2.0 * wt[i] * (d1 + d2);
        }
        dev
    }

    fn initialize(&self, y: &ArrayView1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.5) / 2.0)
    }

    fn validmu(&self, mu: &ArrayView1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0 && m < 1.0)
    }

    fn valideta(&self, _eta: &ArrayView1<f64>) -> bool {
        true
    }
}

/// Poisson family with log link
pub struct Poisson;

impl Family for Poisson {
    fn link(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.ln())
    }

    fn linkinv(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        eta.mapv(|e| e.exp())
    }

    fn mu_eta(&self, eta: &ArrayView1<f64>) -> Array1<f64> {
        eta.mapv(|e| e.exp())
    }

    fn variance(&self, mu: &ArrayView1<f64>) -> Array1<f64> {
        mu.to_owned()
    }

    fn dev_resids(&self, y: &ArrayView1<f64>, mu: &ArrayView1<f64>, wt: &ArrayView1<f64>) -> Array1<f64> {
        let mut dev = Array1::zeros(y.len());
        for i in 0..y.len() {
            let yi = y[i];
            let mui = mu[i].max(1e-10);

            let d = if yi > 0.0 {
                2.0 * wt[i] * (yi * (yi / mui).ln() - (yi - mui))
            } else {
                2.0 * wt[i] * mui
            };
            dev[i] = d;
        }
        dev
    }

    fn initialize(&self, y: &ArrayView1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(1e-10))
    }

    fn validmu(&self, mu: &ArrayView1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }

    fn valideta(&self, _eta: &ArrayView1<f64>) -> bool {
        true
    }
}
