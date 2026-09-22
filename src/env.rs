use crate::rand::Rng;

pub struct Step {
    pub obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
}

pub trait Env {
    fn n_actions(&self) -> usize;
    fn obs_dim(&self) -> usize;
    fn reset(&mut self, rng: &mut Rng) -> Vec<f32>;
    fn step(&mut self, action: usize, rng: &mut Rng) -> Step;
}

const DX: [i32; 4] = [0, 1, 0, -1];
const DY: [i32; 4] = [-1, 0, 1, 0];

pub struct Snake {
    pub w: usize,
    pub h: usize,
    pub max_steps: usize,
    body: Vec<(i32, i32)>,
    dir: usize,
    food: (i32, i32),
    steps: usize,
}

impl Snake {
    pub fn new(w: usize, h: usize, max_steps: usize) -> Self {
        Self {
            w,
            h,
            max_steps,
            body: Vec::new(),
            dir: 1,
            food: (0, 0),
            steps: 0,
        }
    }

    fn empty_cell(&self, rng: &mut Rng) -> (i32, i32) {
        loop {
            let x = rng.below(self.w) as i32;
            let y = rng.below(self.h) as i32;
            if !self.body.contains(&(x, y)) {
                return (x, y);
            }
        }
    }

    fn obs(&self) -> Vec<f32> {
        let mut o = vec![0.0; self.w * self.h + 4];
        for (i, &(x, y)) in self.body.iter().enumerate() {
            o[(y as usize) * self.w + x as usize] = if i == 0 { 1.0 } else { 0.5 };
        }
        o[(self.food.1 as usize) * self.w + self.food.0 as usize] = -1.0;
        o[self.w * self.h + self.dir] = 1.0;
        o
    }
}

impl Env for Snake {
    fn n_actions(&self) -> usize {
        3
    }

    fn obs_dim(&self) -> usize {
        self.w * self.h + 4
    }

    fn reset(&mut self, rng: &mut Rng) -> Vec<f32> {
        let cx = (self.w / 2) as i32;
        let cy = (self.h / 2) as i32;
        self.body = vec![(cx, cy), (cx - 1, cy)];
        self.dir = 1;
        self.steps = 0;
        self.food = self.empty_cell(rng);
        self.obs()
    }

    fn step(&mut self, action: usize, rng: &mut Rng) -> Step {
        assert!(action < 3);
        self.dir = match action {
            0 => self.dir,
            1 => (self.dir + 3) % 4,
            _ => (self.dir + 1) % 4,
        };
        self.steps += 1;
        let (hx, hy) = self.body[0];
        let nx = hx + DX[self.dir];
        let ny = hy + DY[self.dir];
        if nx < 0 || ny < 0 || nx >= self.w as i32 || ny >= self.h as i32 {
            return Step {
                obs: self.obs(),
                reward: -10.0,
                done: true,
            };
        }
        if self.body.contains(&(nx, ny)) {
            return Step {
                obs: self.obs(),
                reward: -10.0,
                done: true,
            };
        }
        self.body.insert(0, (nx, ny));
        if (nx, ny) == self.food {
            if self.body.len() == self.w * self.h {
                return Step {
                    obs: self.obs(),
                    reward: 10.0,
                    done: true,
                };
            }
            self.food = self.empty_cell(rng);
            return Step {
                obs: self.obs(),
                reward: 10.0,
                done: false,
            };
        }
        self.body.pop();
        if self.steps >= self.max_steps {
            return Step {
                obs: self.obs(),
                reward: -1.0,
                done: true,
            };
        }
        Step {
            obs: self.obs(),
            reward: -0.02,
            done: false,
        }
    }
}

pub struct Bandit {
    pub means: Vec<f32>,
    pub noise: f32,
    last: usize,
}

impl Bandit {
    pub fn new(means: Vec<f32>, noise: f32) -> Self {
        Self {
            means,
            noise,
            last: 0,
        }
    }
}

impl Env for Bandit {
    fn n_actions(&self) -> usize {
        self.means.len()
    }

    fn obs_dim(&self) -> usize {
        1
    }

    fn reset(&mut self, _rng: &mut Rng) -> Vec<f32> {
        vec![1.0]
    }

    fn step(&mut self, action: usize, rng: &mut Rng) -> Step {
        self.last = action;
        let r = self.means[action] + rng.gaussian() * self.noise;
        Step {
            obs: vec![1.0],
            reward: r,
            done: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_reset_valid() {
        let mut rng = Rng::new(1);
        let mut s = Snake::new(8, 8, 200);
        let o = s.reset(&mut rng);
        assert_eq!(o.len(), 68);
        assert_eq!(s.body.len(), 2);
        assert!(!s.body.contains(&s.food));
    }

    #[test]
    fn wall_kills() {
        let mut rng = Rng::new(1);
        let mut s = Snake::new(4, 4, 200);
        s.reset(&mut rng);
        s.body = vec![(3, 1), (2, 1)];
        s.dir = 1;
        let st = s.step(0, &mut rng);
        assert!(st.done);
        assert_eq!(st.reward, -10.0);
    }

    #[test]
    fn eating_grows() {
        let mut rng = Rng::new(1);
        let mut s = Snake::new(6, 6, 200);
        s.reset(&mut rng);
        s.body = vec![(2, 2), (1, 2)];
        s.dir = 1;
        s.food = (3, 2);
        let st = s.step(0, &mut rng);
        assert!(!st.done);
        assert_eq!(st.reward, 10.0);
        assert_eq!(s.body.len(), 3);
    }

    #[test]
    fn self_collision_kills() {
        let mut rng = Rng::new(1);
        let mut s = Snake::new(6, 6, 200);
        s.reset(&mut rng);
        s.body = vec![(2, 2), (2, 3), (3, 3), (3, 2)];
        s.dir = 2;
        let st = s.step(0, &mut rng);
        assert!(st.done);
    }
}
