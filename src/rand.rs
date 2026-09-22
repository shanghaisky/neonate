#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1).wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1),
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x.max(1);
        x
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / ((1u32 << 24) as f32)
    }

    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    pub fn gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let (mut u, mut v, mut s) = (0.0, 0.0, 0.0);
        while s >= 1.0 || s == 0.0 {
            u = self.range_f32(-1.0, 1.0);
            v = self.range_f32(-1.0, 1.0);
            s = u * u + v * v;
        }
        let m = (-2.0 * s.ln() / s).sqrt();
        self.spare = Some(v * m);
        u * m
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i + 1));
        }
    }

    pub fn choice<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn gaussian_stats() {
        let mut r = Rng::new(7);
        let n = 20000;
        let (mut m, mut v) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let x = r.gaussian() as f64;
            m += x;
            v += x * x;
        }
        m /= n as f64;
        v = v / n as f64 - m * m;
        assert!(m.abs() < 0.05, "mean {m}");
        assert!((v - 1.0).abs() < 0.05, "var {v}");
    }

    #[test]
    fn uniform_bounds() {
        let mut r = Rng::new(1);
        for _ in 0..10000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
