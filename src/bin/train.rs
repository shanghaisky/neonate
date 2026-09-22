use neonate::{Dqn, Env, Rng, Snake};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let out = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("snake-dqn.bin"));

    let mut rng = Rng::new(seed);
    let mut env = Snake::new(8, 8, 150);
    let mut dqn = Dqn::new(&mut rng, env.obs_dim(), env.n_actions(), &[128, 64], 3e-4);
    dqn.eps_decay = 0.9992;

    let mut recent: Vec<f32> = Vec::with_capacity(100);
    for ep in 1..=episodes {
        let mut obs = env.reset(&mut rng);
        let mut food = 0;
        loop {
            let a = dqn.act(&obs, &mut rng);
            let st = env.step(a, &mut rng);
            if st.reward > 1.0 {
                food += 1;
            }
            dqn.remember(obs, a, st.reward, st.obs.clone(), st.done);
            dqn.train_step(&mut rng);
            dqn.decay();
            obs = st.obs;
            if st.done {
                break;
            }
        }
        recent.push(food as f32);
        if recent.len() > 100 {
            recent.remove(0);
        }
        if ep % 100 == 0 {
            let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            println!("ep {ep:>5}  food/100ep {avg:5.2}  eps {:.3}", dqn.epsilon);
        }
    }
    dqn.q.save(&out).expect("save weights");
    println!("saved {}", out.display());
}
