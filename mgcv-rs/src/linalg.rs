/// Basic linear algebra operations for MGCV
use std::ops::{Add, Sub, Mul};

#[derive(Clone, Debug)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(n: usize) -> Self {
        Self {
            data: vec![0.0; n],
        }
    }

    pub fn ones(n: usize) -> Self {
        Self {
            data: vec![1.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
            data.push(start + t * (end - start));
        }
        Self { data }
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        assert_eq!(self.len(), other.len());
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mapv<F>(&self, f: F) -> Vector
    where
        F: Fn(f64) -> f64,
    {
        Vector {
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }
}

impl Add for &Vector {
    type Output = Vector;

    fn add(self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        Vector {
            data: self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl Sub for &Vector {
    type Output = Vector;

    fn sub(self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len());
        Vector {
            data: self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        Vector {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f64>, nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }

    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrows && j < self.ncols);
        self.data[i * self.ncols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        assert!(i < self.nrows && j < self.ncols);
        self.data[i * self.ncols + j] = value;
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.ncols, self.nrows);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.ncols, other.nrows);
        let mut result = Matrix::zeros(self.nrows, other.ncols);

        for i in 0..self.nrows {
            for j in 0..other.ncols {
                let mut sum = 0.0;
                for k in 0..self.ncols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn dot_vec(&self, vec: &Vector) -> Vector {
        assert_eq!(self.ncols, vec.len());
        let mut result = Vector::zeros(self.nrows);

        for i in 0..self.nrows {
            let mut sum = 0.0;
            for j in 0..self.ncols {
                sum += self.get(i, j) * vec.data[j];
            }
            result.data[i] = sum;
        }
        result
    }

    pub fn diag(&self) -> Vector {
        let n = self.nrows.min(self.ncols);
        let mut result = Vector::zeros(n);
        for i in 0..n {
            result.data[i] = self.get(i, i);
        }
        result
    }

    /// Cholesky decomposition: A = L * L^T
    pub fn cholesky(&self) -> Option<Matrix> {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        let mut L = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += L.get(i, k) * L.get(j, k);
                }

                if i == j {
                    let val = self.get(i, i) - sum;
                    if val <= 0.0 {
                        return None; // Not positive definite
                    }
                    L.set(i, j, val.sqrt());
                } else {
                    let l_jj = L.get(j, j);
                    if l_jj.abs() < 1e-10 {
                        return None;
                    }
                    L.set(i, j, (self.get(i, j) - sum) / l_jj);
                }
            }
        }
        Some(L)
    }

    /// Solve L * x = b where L is lower triangular
    fn solve_lower_triangular(&self, b: &Vector) -> Vector {
        let n = self.nrows;
        let mut x = Vector::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += self.get(i, j) * x.data[j];
            }
            x.data[i] = (b.data[i] - sum) / self.get(i, i);
        }
        x
    }

    /// Solve U * x = b where U is upper triangular
    fn solve_upper_triangular(&self, b: &Vector) -> Vector {
        let n = self.nrows;
        let mut x = Vector::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += self.get(i, j) * x.data[j];
            }
            x.data[i] = (b.data[i] - sum) / self.get(i, i);
        }
        x
    }

    /// Solve A * x = b using Cholesky decomposition
    pub fn solve(&self, b: &Vector) -> Option<Vector> {
        let L = self.cholesky()?;
        let Lt = L.transpose();

        // Solve L * y = b
        let y = L.solve_lower_triangular(b);

        // Solve L^T * x = y
        let x = Lt.solve_upper_triangular(&y);

        Some(x)
    }

    /// Compute determinant using Cholesky decomposition
    pub fn determinant(&self) -> Option<f64> {
        let L = self.cholesky()?;
        let mut det = 1.0;
        for i in 0..L.nrows {
            det *= L.get(i, i);
        }
        Some(det * det) // det(A) = det(L * L^T) = det(L)^2
    }

    /// Simple matrix inversion using Cholesky (for positive definite matrices)
    pub fn inverse(&self) -> Option<Matrix> {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        let mut inv = Matrix::zeros(n, n);

        // Solve A * X = I column by column
        for j in 0..n {
            let mut b = Vector::zeros(n);
            b.data[j] = 1.0;

            let x = self.solve(&b)?;
            for i in 0..n {
                inv.set(i, j, x.data[i]);
            }
        }

        Some(inv)
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.nrows, other.nrows);
        assert_eq!(self.ncols, other.ncols);

        Matrix {
            data: self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Matrix {
        Matrix {
            data: self.data.iter().map(|x| x * scalar).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}
