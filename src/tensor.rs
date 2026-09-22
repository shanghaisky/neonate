use crate::rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Array {
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; n],
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn randn(rng: &mut Rng, shape: &[usize], scale: f32) -> Self {
        let n = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: (0..n).map(|_| rng.gaussian() * scale).collect(),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rows(&self) -> usize {
        assert!(self.shape.len() == 2);
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        assert!(self.shape.len() == 2);
        self.shape[1]
    }

    pub fn matmul(&self, o: &Array) -> Array {
        assert!(self.shape.len() == 2 && o.shape.len() == 2);
        assert_eq!(self.cols(), o.rows());
        let (m, k, n) = (self.rows(), self.cols(), o.cols());
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a * o.data[p * n + j];
                }
            }
        }
        Array {
            shape: vec![m, n],
            data: out,
        }
    }

    pub fn transpose(&self) -> Array {
        assert!(self.shape.len() == 2);
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Array {
            shape: vec![n, m],
            data: out,
        }
    }

    pub fn add(&self, o: &Array) -> Array {
        assert_eq!(self.shape, o.shape);
        Array {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(o.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn add_row_bias(&self, bias: &Array) -> Array {
        assert!(self.shape.len() == 2);
        assert_eq!(bias.shape, vec![self.cols()]);
        let (m, n) = (self.rows(), self.cols());
        let mut out = self.data.clone();
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] += bias.data[j];
            }
        }
        Array {
            shape: self.shape.clone(),
            data: out,
        }
    }

    pub fn sum_rows(&self) -> Array {
        assert!(self.shape.len() == 2);
        let n = self.cols();
        let mut out = vec![0.0; n];
        for row in self.data.chunks_exact(n) {
            for (j, &v) in row.iter().enumerate() {
                out[j] += v;
            }
        }
        Array {
            shape: vec![n],
            data: out,
        }
    }

    pub fn relu(&self) -> Array {
        self.map(|x| x.max(0.0))
    }

    pub fn tanh(&self) -> Array {
        self.map(|x| x.tanh())
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Array {
        Array {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    pub fn sub(&self, o: &Array) -> Array {
        assert_eq!(self.shape, o.shape);
        Array {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(o.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    pub fn scale(&self, s: f32) -> Array {
        self.map(|x| x * s)
    }

    pub fn square(&self) -> Array {
        self.map(|x| x * x)
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn max_abs(&self) -> f32 {
        self.data.iter().map(|x| x.abs()).fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_known() {
        let a = Array::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Array::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = Array::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn bias_and_sum_rows() {
        let a = Array::from_vec(&[2, 2], vec![1., 2., 3., 4.]);
        let b = Array::from_vec(&[2], vec![10., 20.]);
        let c = a.add_row_bias(&b);
        assert_eq!(c.data, vec![11., 22., 13., 24.]);
        assert_eq!(c.sum_rows().data, vec![24., 46.]);
    }
}
