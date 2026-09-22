# Neonate

**Reinforcement learning from scratch in Rust. Zero dependencies.**

A reverse-mode autograd engine, multilayer perceptrons with Adam, a DQN agent with experience replay and target networks, and game environments — one crate, no crates. It learns: a Snake agent goes from 0.26 to 1.44 food per episode over 400 training episodes.

```
ep   100  food/100ep  0.26  eps 0.214
ep   200  food/100ep  0.91  eps 0.050
ep   300  food/100ep  1.45  eps 0.050
ep   400  food/100ep  1.44  eps 0.050
```

## What's inside

| Module | What it is |
|---|---|
| `tensor.rs` | Dense f32 arrays: matmul, transpose, broadcasting bias, activations |
| `autograd.rs` | Tape-based reverse-mode autodiff (MatMul, Add, ReLU, Tanh, MSE), gradient-checked against finite differences |
| `nn.rs` | Kaiming-initialized MLPs, Adam with bias correction, gradient clipping, binary save/load |
| `env.rs` | `Env` trait, `Snake` (relative steering, shaped rewards), `Bandit` (fast deterministic tests) |
| `dqn.rs` | DQN: replay buffer, epsilon-greedy schedule, target network sync, Huber-style clipped updates |
| `rand.rs` | Seeded xorshift RNG with Gaussian sampling — every run reproducible |

## Requirements

- Rust stable toolchain (built with 1.98.1; any recent stable works)
- A C linker (`cc`), bundled with every standard Rust installation
- Nothing else: zero crates, `std` only

## Quickstart

```bash
cargo test                 # 17 tests: gradchecks, regression learning, bandit learning, env rules
cargo run --release --bin train [episodes] [seed] [weights-out]
cargo run --release --bin play [weights] [episodes]
```

Watch it play Snake in your terminal (ASCII render, greedy policy from saved weights).

## API sketch

```rust
use neonate::{Dqn, Env, Rng, Snake};

let mut rng = Rng::new(7);
let mut env = Snake::new(8, 8, 150);
let mut dqn = Dqn::new(&mut rng, env.obs_dim(), env.n_actions(), &[128, 64], 3e-4);

let mut obs = env.reset(&mut rng);
loop {
    let a = dqn.act(&obs, &mut rng);
    let st = env.step(a, &mut rng);
    dqn.remember(obs, a, st.reward, st.obs.clone(), st.done);
    dqn.train_step(&mut rng);
    dqn.decay();
    obs = st.obs;
    if st.done { break; }
}
```

## Testing

- `autograd`: analytic gradients vs finite differences on MLP and tanh graphs
- `nn`: MLP fits `y = 2x + 1` (loss drops 50x), save/load roundtrip
- `dqn`: bandit test proves the agent finds the best arm (avg reward > 0.55 vs 0.5 random)
- `env`: wall/self collision, growth on eat, valid resets
- `tests/snake_smoke.rs`: full Snake+DQN loop runs and eats

## Roadmap

- [ ] Transformer + BPE tokenizer on top of this autograd engine (tiny LLM)
- [ ] PPO alongside DQN
- [ ] More environments (gridworlds, cart-pole)
- [ ] Prioritized experience replay, dueling heads

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).
