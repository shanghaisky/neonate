use neonate::{Dqn, Env, Rng, Snake};

#[test]
fn snake_dqn_loop_runs_and_eats() {
    let mut rng = Rng::new(2026);
    let mut env = Snake::new(5, 5, 60);
    let mut dqn = Dqn::new(&mut rng, env.obs_dim(), env.n_actions(), &[32], 1e-3);
    dqn.eps_decay = 0.99;
    let mut total_food = 0;
    let mut losses = 0;
    for _ in 0..60 {
        let mut obs = env.reset(&mut rng);
        loop {
            let a = dqn.act(&obs, &mut rng);
            let st = env.step(a, &mut rng);
            if st.reward > 1.0 {
                total_food += 1;
            }
            dqn.remember(obs, a, st.reward, st.obs.clone(), st.done);
            if dqn.train_step(&mut rng).is_some() {
                losses += 1;
            }
            dqn.decay();
            obs = st.obs;
            if st.done {
                break;
            }
        }
    }
    assert!(losses > 0, "should have trained");
    assert!(
        total_food > 0,
        "random play should stumble into food sometimes"
    );
}
