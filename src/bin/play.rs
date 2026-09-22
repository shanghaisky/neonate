use neonate::{Env, Mlp, Rng, Snake};
use std::path::PathBuf;

fn render(env_obs_grid: &[f32], w: usize, h: usize, score: u32) {
    println!("score {score}");
    for y in 0..h {
        for x in 0..w {
            let v = env_obs_grid[y * w + x];
            let c = if v == 1.0 {
                '#'
            } else if v == 0.5 {
                'o'
            } else if v == -1.0 {
                '*'
            } else {
                '.'
            };
            print!("{c}");
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("snake-dqn.bin"));
    let episodes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let net = Mlp::load(&weights).expect("load weights (run `train` first)");
    let mut rng = Rng::new(99);
    let mut env = Snake::new(8, 8, 150);
    for _ in 0..episodes {
        let mut obs = env.reset(&mut rng);
        let mut score = 0;
        loop {
            print!("\x1b[2J\x1b[H");
            render(&obs, 8, 8, score);
            let q = net.predict(&neonate::Array::from_vec(&[1, obs.len()], obs.clone()));
            let mut a = 0;
            for i in 1..q.data.len() {
                if q.data[i] > q.data[a] {
                    a = i;
                }
            }
            let st = env.step(a, &mut rng);
            if st.reward > 1.0 {
                score += 1;
            }
            obs = st.obs;
            std::thread::sleep(std::time::Duration::from_millis(90));
            if st.done {
                print!("\x1b[2J\x1b[H");
                render(&obs, 8, 8, score);
                println!("died with score {score}");
                break;
            }
        }
    }
}
