use crate::autograd::Tape;
use crate::nn::{clip_grads, Adam, Mlp};
use crate::rand::Rng;
use crate::tensor::Array;

#[derive(Clone)]
struct Transition {
    s: Vec<f32>,
    a: usize,
    r: f32,
    s2: Vec<f32>,
    done: bool,
}

struct Replay {
    buf: Vec<Transition>,
    cap: usize,
}

impl Replay {
    fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap.min(1024)),
            cap,
        }
    }

    fn push(&mut self, t: Transition) {
        if self.buf.len() == self.cap {
            self.buf.remove(0);
        }
        self.buf.push(t);
    }

    fn sample(&self, rng: &mut Rng, n: usize) -> Vec<Transition> {
        (0..n)
            .map(|_| self.buf[rng.below(self.buf.len())].clone())
            .collect()
    }
}

pub struct Dqn {
    pub q: Mlp,
    pub target: Mlp,
    pub opt: Adam,
    replay: Replay,
    pub gamma: f32,
    pub epsilon: f32,
    pub eps_min: f32,
    pub eps_decay: f32,
    pub reward_scale: f32,
    target_sync: u64,
    pub steps: u64,
    batch: usize,
    obs_dim: usize,
    n_actions: usize,
}

impl Dqn {
    pub fn new(rng: &mut Rng, obs_dim: usize, n_actions: usize, hidden: &[usize], lr: f32) -> Self {
        let mut sizes = vec![obs_dim];
        sizes.extend_from_slice(hidden);
        sizes.push(n_actions);
        let q = Mlp::new(rng, &sizes);
        let target = q.zeros_like();
        let mut d = Self {
            q,
            target,
            opt: Adam::new(lr),
            replay: Replay::new(50_000),
            gamma: 0.99,
            epsilon: 1.0,
            eps_min: 0.05,
            eps_decay: 0.9995,
            reward_scale: 0.1,
            target_sync: 500,
            steps: 0,
            batch: 64,
            obs_dim,
            n_actions,
        };
        d.target.copy_from(&d.q);
        d
    }

    pub fn greedy(&self, obs: &[f32]) -> usize {
        let q = self
            .q
            .predict(&Array::from_vec(&[1, self.obs_dim], obs.to_vec()));
        argmax(&q.data)
    }

    pub fn act(&self, obs: &[f32], rng: &mut Rng) -> usize {
        if rng.next_f32() < self.epsilon {
            rng.below(self.n_actions)
        } else {
            self.greedy(obs)
        }
    }

    pub fn remember(&mut self, s: Vec<f32>, a: usize, r: f32, s2: Vec<f32>, done: bool) {
        self.replay.push(Transition { s, a, r, s2, done });
    }

    pub fn decay(&mut self) {
        self.epsilon = (self.epsilon * self.eps_decay).max(self.eps_min);
    }

    pub fn train_step(&mut self, rng: &mut Rng) -> Option<f32> {
        if self.replay.buf.len() < self.batch {
            return None;
        }
        let batch = self.replay.sample(rng, self.batch);
        let b = batch.len();
        let mut s_data = Vec::with_capacity(b * self.obs_dim);
        let mut s2_data = Vec::with_capacity(b * self.obs_dim);
        for t in &batch {
            s_data.extend_from_slice(&t.s);
            s2_data.extend_from_slice(&t.s2);
        }
        let s = Array::from_vec(&[b, self.obs_dim], s_data);
        let s2 = Array::from_vec(&[b, self.obs_dim], s2_data);

        let tq = self.target.predict(&s2);
        let pq = self.q.predict(&s);
        let mut y = pq.data.clone();
        for (i, t) in batch.iter().enumerate() {
            let mut best = f32::NEG_INFINITY;
            for a in 0..self.n_actions {
                best = best.max(tq.data[i * self.n_actions + a]);
            }
            let target = t.r * self.reward_scale + if t.done { 0.0 } else { self.gamma * best };
            y[i * self.n_actions + t.a] = target;
        }
        let y = Array::from_vec(&[b, self.n_actions], y);

        let mut tape = Tape::new();
        let xi = tape.input(s);
        let (o, ids) = self.q.forward(&mut tape, xi);
        let l = tape.mse(o, &y);
        let lv = tape.value(l).data[0];
        tape.backward(l);
        clip_grads(&mut tape, &ids, 10.0);
        self.opt.step(&tape, &ids, &mut self.q.params);

        self.steps += 1;
        if self.steps.is_multiple_of(self.target_sync) {
            self.target.copy_from(&self.q);
        }
        Some(lv)
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[b] {
            b = i;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{Bandit, Env};

    #[test]
    fn bandit_learns_best_arm() {
        let mut rng = Rng::new(1234);
        let mut env = Bandit::new(vec![0.2, 0.8], 0.1);
        let mut dqn = Dqn::new(&mut rng, 1, 2, &[16], 0.01);
        dqn.eps_decay = 0.995;
        dqn.eps_min = 0.05;
        for _ in 0..600 {
            let s = env.reset(&mut rng);
            let a = dqn.act(&s, &mut rng);
            let st = env.step(a, &mut rng);
            dqn.remember(s, a, st.reward, st.obs, st.done);
            dqn.train_step(&mut rng);
            dqn.decay();
        }
        let mut sum = 0.0;
        for _ in 0..200 {
            let s = env.reset(&mut rng);
            let a = dqn.greedy(&s);
            sum += env.step(a, &mut rng).reward;
        }
        let avg = sum / 200.0;
        assert!(
            avg > 0.55,
            "greedy avg reward {avg}, should favor the 0.8 arm"
        );
    }
}
