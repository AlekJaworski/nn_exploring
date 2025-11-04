use std::fmt;

#[derive(Debug)]
pub enum MgcvError {
    LinAlgError(String),
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    ConvergenceError(String),
    InvalidParameter(String),
    OptimizationError(String),
    InvalidSmooth(String),
}

impl fmt::Display for MgcvError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MgcvError::LinAlgError(msg) => write!(f, "Linear algebra error: {}", msg),
            MgcvError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            MgcvError::ConvergenceError(msg) => write!(f, "Convergence failed: {}", msg),
            MgcvError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            MgcvError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            MgcvError::InvalidSmooth(msg) => write!(f, "Invalid smooth specification: {}", msg),
        }
    }
}

impl std::error::Error for MgcvError {}

pub type Result<T> = std::result::Result<T, MgcvError>;
