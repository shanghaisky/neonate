use crate::autograd::{Id, Tape};
use crate::rand::Rng;
use crate::tensor::Array;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct Mlp {
    pub sizes: Vec<usize>,
    pub params: Vec<Array>,
}

impl Mlp {
    pub fn new(rng: &mut Rng, sizes: &[usize]) -> Self {
        assert!(sizes.len() >= 2);
        let mut params = Vec::new();
        for w in sizes.windows(2) {
            let (din, dout) = (w[0], w[1]);
            let s = (2.0 / din as f32).sqrt();
            params.push(Array::randn(rng, &[din, dout], s));
            params.push(Array::zeros(&[dout]));
        }
        Self {
            sizes: sizes.to_vec(),
            params,
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self {
            sizes: self.sizes.clone(),
            params: self.params.iter().map(|p| Array::zeros(&p.shape)).collect(),
        }
    }

    pub fn forward(&self, tape: &mut Tape, x: Id) -> (Id, Vec<Id>) {
        let mut ids = Vec::with_capacity(self.params.len());
        let mut h = x;
        let n = self.params.len();
        for (i, p) in self.params.iter().enumerate() {
            let pid = tape.param(p.clone());
            ids.push(pid);
            if i % 2 == 0 {
                h = tape.matmul(h, pid);
            } else {
                h = tape.add_bias(h, pid);
                if i + 1 < n {
                    h = tape.relu(h);
                }
            }
        }
        (h, ids)
    }

    pub fn predict(&self, x: &Array) -> Array {
        let mut tape = Tape::new();
        let xi = tape.input(x.clone());
        let (o, _) = self.forward(&mut tape, xi);
        tape.value(o).clone()
    }

    pub fn copy_from(&mut self, o: &Mlp) {
        assert_eq!(self.sizes, o.sizes);
        for (a, b) in self.params.iter_mut().zip(o.params.iter()) {
            *a = b.clone();
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(&(self.params.len() as u64).to_le_bytes())?;
        for p in &self.params {
            f.write_all(&(p.shape.len() as u64).to_le_bytes())?;
            for d in &p.shape {
                f.write_all(&(*d as u64).to_le_bytes())?;
            }
            for v in &p.data {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut u = [0u8; 8];
        let rd = |f: &mut File, u: &mut [u8; 8]| -> std::io::Result<u64> {
            f.read_exact(u)?;
            Ok(u64::from_le_bytes(*u))
        };
        let n = rd(&mut f, &mut u)? as usize;
        let mut params = Vec::with_capacity(n);
        let mut sizes = Vec::new();
        for i in 0..n {
            let nd = rd(&mut f, &mut u)? as usize;
            let mut shape = Vec::with_capacity(nd);
            for _ in 0..nd {
                shape.push(rd(&mut f, &mut u)? as usize);
            }
            let m: usize = shape.iter().product();
            let mut data = vec![0.0; m];
            for v in data.iter_mut() {
                let mut b = [0u8; 4];
                f.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }
            if i % 2 == 0 && nd == 2 {
                if sizes.is_empty() {
                    sizes.push(shape[0]);
                }
                sizes.push(shape[1]);
            }
            params.push(Array { shape, data });
        }
        Ok(Self { sizes, params })
    }
}

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: u64,
    m: Vec<Array>,
    v: Vec<Array>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, tape: &Tape, ids: &[Id], params: &mut [Array]) {
        assert_eq!(ids.len(), params.len());
        if self.m.is_empty() {
            self.m = params.iter().map(|p| Array::zeros(&p.shape)).collect();
            self.v = params.iter().map(|p| Array::zeros(&p.shape)).collect();
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = match tape.grad(ids[i]) {
                Some(g) => g,
                None => continue,
            };
            for j in 0..params[i].data.len() {
                let gj = g.data[j];
                self.m[i].data[j] = self.beta1 * self.m[i].data[j] + (1.0 - self.beta1) * gj;
                self.v[i].data[j] = self.beta2 * self.v[i].data[j] + (1.0 - self.beta2) * gj * gj;
                let mh = self.m[i].data[j] / bc1;
                let vh = self.v[i].data[j] / bc2;
                params[i].data[j] -= self.lr * mh / (vh.sqrt() + self.eps);
                if !params[i].data[j].is_finite() {
                    params[i].data[j] = 0.0;
                }
            }
        }
    }
}

pub fn clip_grads(tape: &mut Tape, ids: &[Id], max_norm: f32) {
    let n = tape.grad_norm(ids);
    if n > max_norm && n > 0.0 {
        tape.scale_grads(ids, max_norm / n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlp_learns_line() {
        let mut rng = Rng::new(5);
        let mut net = Mlp::new(&mut rng, &[1, 16, 1]);
        let mut opt = Adam::new(0.02);
        let xs: Vec<f32> = (-20..20).map(|i| i as f32 / 20.0).collect();
        let x = Array::from_vec(&[40, 1], xs.clone());
        let t = Array::from_vec(&[40, 1], xs.iter().map(|v| 2.0 * v + 1.0).collect());
        let mut first = 0.0;
        let mut last = 0.0;
        for step in 0..300 {
            let mut tape = Tape::new();
            let xi = tape.input(x.clone());
            let (o, ids) = net.forward(&mut tape, xi);
            let l = tape.mse(o, &t);
            let lv = tape.value(l).data[0];
            if step == 0 {
                first = lv;
            }
            last = lv;
            tape.backward(l);
            opt.step(&tape, &ids, &mut net.params);
        }
        assert!(last < first * 0.02, "loss {first} -> {last}");
        let p = net.predict(&Array::from_vec(&[1, 1], vec![0.5]));
        assert!((p.data[0] - 2.0).abs() < 0.15, "pred {}", p.data[0]);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut rng = Rng::new(9);
        let net = Mlp::new(&mut rng, &[4, 8, 2]);
        let p = std::env::temp_dir().join(format!("neonate-w-{}.bin", std::process::id()));
        net.save(&p).unwrap();
        let back = Mlp::load(&p).unwrap();
        assert_eq!(back.sizes, net.sizes);
        assert_eq!(back.params, net.params);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn copy_from_syncs() {
        let mut rng = Rng::new(2);
        let a = Mlp::new(&mut rng, &[2, 4, 1]);
        let mut b = Mlp::zeros_like(&a);
        b.copy_from(&a);
        assert_eq!(a.params, b.params);
    }
}
