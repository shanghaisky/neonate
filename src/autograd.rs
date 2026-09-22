use crate::tensor::Array;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Id(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Input,
    MatMul,
    AddBias,
    Add,
    Relu,
    Tanh,
    Mse,
}

struct Node {
    kind: Kind,
    inputs: Vec<Id>,
    target: Option<Array>,
}

pub struct Tape {
    values: Vec<Array>,
    grads: Vec<Option<Array>>,
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            grads: Vec::new(),
            nodes: Vec::new(),
        }
    }

    fn push(&mut self, kind: Kind, inputs: Vec<Id>, target: Option<Array>, val: Array) -> Id {
        let id = Id(self.values.len());
        self.values.push(val);
        self.grads.push(None);
        self.nodes.push(Node {
            kind,
            inputs,
            target,
        });
        id
    }

    pub fn input(&mut self, a: Array) -> Id {
        self.push(Kind::Input, vec![], None, a)
    }

    pub fn param(&mut self, a: Array) -> Id {
        self.push(Kind::Input, vec![], None, a)
    }

    pub fn value(&self, id: Id) -> &Array {
        &self.values[id.0]
    }

    pub fn grad(&self, id: Id) -> Option<&Array> {
        self.grads[id.0].as_ref()
    }

    pub fn matmul(&mut self, a: Id, b: Id) -> Id {
        let v = self.values[a.0].matmul(&self.values[b.0]);
        self.push(Kind::MatMul, vec![a, b], None, v)
    }

    pub fn add_bias(&mut self, x: Id, b: Id) -> Id {
        let v = self.values[x.0].add_row_bias(&self.values[b.0]);
        self.push(Kind::AddBias, vec![x, b], None, v)
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        let v = self.values[a.0].add(&self.values[b.0]);
        self.push(Kind::Add, vec![a, b], None, v)
    }

    pub fn relu(&mut self, x: Id) -> Id {
        let v = self.values[x.0].relu();
        self.push(Kind::Relu, vec![x], None, v)
    }

    pub fn tanh(&mut self, x: Id) -> Id {
        let v = self.values[x.0].tanh();
        self.push(Kind::Tanh, vec![x], None, v)
    }

    pub fn mse(&mut self, pred: Id, target: &Array) -> Id {
        let p = &self.values[pred.0];
        assert_eq!(p.shape, target.shape);
        let d: f32 = p
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / p.data.len() as f32;
        let v = Array::from_vec(&[1], vec![d]);
        self.push(Kind::Mse, vec![pred], Some(target.clone()), v)
    }

    pub fn zero_grad(&mut self) {
        for g in self.grads.iter_mut() {
            *g = None;
        }
    }

    pub fn grad_norm(&self, ids: &[Id]) -> f32 {
        let mut s = 0.0;
        for id in ids {
            if let Some(g) = &self.grads[id.0] {
                s += g.data.iter().map(|x| x * x).sum::<f32>();
            }
        }
        s.sqrt()
    }

    pub fn scale_grads(&mut self, ids: &[Id], k: f32) {
        for id in ids {
            if let Some(g) = self.grads[id.0].as_mut() {
                for x in g.data.iter_mut() {
                    *x *= k;
                }
            }
        }
    }

    fn accumulate(&mut self, id: Id, g: Array) {
        match &mut self.grads[id.0] {
            Some(old) => {
                let s = old.add(&g);
                *old = s;
            }
            None => self.grads[id.0] = Some(g),
        }
    }

    pub fn backward(&mut self, loss: Id) {
        self.accumulate(loss, Array::from_vec(&[1], vec![1.0]));
        for i in (0..self.nodes.len()).rev() {
            let g = match self.grads[i].clone() {
                Some(g) => g,
                None => continue,
            };
            let node = &self.nodes[i];
            match node.kind {
                Kind::Input => {}
                Kind::MatMul => {
                    let (a, b) = (node.inputs[0], node.inputs[1]);
                    let ga = g.matmul(&self.values[b.0].transpose());
                    let gb = self.values[a.0].transpose().matmul(&g);
                    self.accumulate(a, ga);
                    self.accumulate(b, gb);
                }
                Kind::AddBias => {
                    let (x, b) = (node.inputs[0], node.inputs[1]);
                    self.accumulate(x, g.clone());
                    self.accumulate(b, g.sum_rows());
                }
                Kind::Add => {
                    let (a, b) = (node.inputs[0], node.inputs[1]);
                    self.accumulate(a, g.clone());
                    self.accumulate(b, g);
                }
                Kind::Relu => {
                    let x = node.inputs[0];
                    let xv = self.values[x.0].clone();
                    let m: Vec<f32> = xv
                        .data
                        .iter()
                        .zip(g.data.iter())
                        .map(|(&v, &d)| if v > 0.0 { d } else { 0.0 })
                        .collect();
                    self.accumulate(x, Array::from_vec(&xv.shape, m));
                }
                Kind::Tanh => {
                    let x = node.inputs[0];
                    let o = self.values[i].clone();
                    let m: Vec<f32> = o
                        .data
                        .iter()
                        .zip(g.data.iter())
                        .map(|(&v, &d)| d * (1.0 - v * v))
                        .collect();
                    self.accumulate(x, Array::from_vec(&o.shape, m));
                }
                Kind::Mse => {
                    let p = node.inputs[0];
                    let target = node.target.clone().unwrap();
                    let pv = self.values[p.0].clone();
                    let n = pv.data.len() as f32;
                    let m: Vec<f32> = pv
                        .data
                        .iter()
                        .zip(target.data.iter())
                        .map(|(&a, &b)| 2.0 * (a - b) / n * g.data[0])
                        .collect();
                    self.accumulate(p, Array::from_vec(&pv.shape, m));
                }
            }
        }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand::Rng;

    fn numeric_grad(f: impl Fn(&Array) -> f32, w: &Array, e: f32) -> Array {
        let mut g = vec![0.0; w.data.len()];
        for i in 0..w.data.len() {
            let mut p = w.data.clone();
            let mut m = w.data.clone();
            p[i] += e;
            m[i] -= e;
            let fp = f(&Array::from_vec(&w.shape, p));
            let fm = f(&Array::from_vec(&w.shape, m));
            g[i] = (fp - fm) / (2.0 * e);
        }
        Array::from_vec(&w.shape, g)
    }

    #[test]
    fn gradcheck_mlp() {
        let mut rng = Rng::new(3);
        let x = Array::randn(&mut rng, &[4, 5], 1.0);
        let w1 = Array::randn(&mut rng, &[5, 6], 0.5);
        let b1 = Array::randn(&mut rng, &[6], 0.5);
        let w2 = Array::randn(&mut rng, &[6, 2], 0.5);
        let t = Array::randn(&mut rng, &[4, 2], 1.0);

        let run = |w1v: &Array| {
            let mut tp = Tape::new();
            let xi = tp.input(x.clone());
            let w = tp.param(w1v.clone());
            let b = tp.param(b1.clone());
            let w2i = tp.param(w2.clone());
            let m = tp.matmul(xi, w);
            let ab = tp.add_bias(m, b);
            let h = tp.relu(ab);
            let o = tp.matmul(h, w2i);
            let l = tp.mse(o, &t);
            tp.value(l).data[0]
        };
        let num = numeric_grad(run, &w1, 1e-3);

        let mut tp = Tape::new();
        let xi = tp.input(x);
        let w = tp.param(w1.clone());
        let b = tp.param(b1);
        let w2i = tp.param(w2);
        let m = tp.matmul(xi, w);
        let ab = tp.add_bias(m, b);
        let h = tp.relu(ab);
        let o = tp.matmul(h, w2i);
        let l = tp.mse(o, &t);
        tp.backward(l);
        let ana = tp.grad(w).unwrap();
        for (a, n) in ana.data.iter().zip(num.data.iter()) {
            assert!((a - n).abs() < 1e-2, "analytic {a} vs numeric {n}");
        }
    }

    #[test]
    fn gradcheck_tanh_add() {
        let mut rng = Rng::new(11);
        let a = Array::randn(&mut rng, &[3, 4], 1.0);
        let b = Array::randn(&mut rng, &[3, 4], 1.0);
        let run = |av: &Array| {
            let mut tp = Tape::new();
            let x = tp.param(av.clone());
            let y = tp.param(b.clone());
            let ad = tp.add(x, y);
            let o = tp.tanh(ad);
            let zero = Array::from_vec(&[3, 4], vec![0.0; 12]);
            let l = tp.mse(o, &zero);
            tp.value(l).data[0]
        };
        let num = numeric_grad(run, &a, 1e-3);
        let mut tp = Tape::new();
        let x = tp.param(a.clone());
        let y = tp.param(b);
        let ad = tp.add(x, y);
        let o = tp.tanh(ad);
        let zero = Array::from_vec(&[3, 4], vec![0.0; 12]);
        let l = tp.mse(o, &zero);
        tp.backward(l);
        let ana = tp.grad(x).unwrap();
        for (x_, n) in ana.data.iter().zip(num.data.iter()) {
            assert!((x_ - n).abs() < 1e-2, "analytic {x_} vs numeric {n}");
        }
    }
}
