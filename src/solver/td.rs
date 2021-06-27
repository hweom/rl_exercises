use crate::solver::*;

pub fn find_action_values_expected_sarsa<S, A, StartState, RandomAction, NextState>(
    start_state: &StartState,
    random_action: &RandomAction,
    next_state: &NextState,
    discount: f64,
    exploration_fraction: f64,
    alpha: f64,
    iterations: u64,
) -> HashMap<S, HashMap<A, f64>>
where
    S: Eq + Hash + Debug + Clone,
    A: Eq + Hash + Debug + Clone,
    StartState: Fn() -> S,
    RandomAction: Fn(&S) -> A,
    NextState: Fn(&S, &A) -> (Option<S>, f64),
{
    let mut action_values: HashMap<S, HashMap<A, f64>> = HashMap::new();

    for _ in 0..iterations {
        // Generate a single episode.
        let mut state = start_state();
        loop {
            // Determine the next action using ε-greedy policy from Q.
            let action = {
                let maybe_state_action_values = action_values.get(&state);
                // Pick greedy action if available and failing "exploration" probability check.
                // Otherwise, choose at random from all available actions.
                if maybe_state_action_values.is_some()
                    && rand::random::<f64>() >= exploration_fraction
                {
                    let state_action_values = maybe_state_action_values.unwrap();

                    // Find the maximum action value.
                    let max_value = state_action_values
                        .iter()
                        .map(|(_, v)| v)
                        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));

                    // Find the actions with max action value (can be multiple!).
                    let greedy_actions: Vec<A> = state_action_values
                        .iter()
                        .filter(|(_, v)| (*v - max_value).abs() < 1e-6)
                        .map(|(a, v)| a.clone())
                        .collect();

                    assert!(greedy_actions.len() > 0);
                    if greedy_actions.len() == 1 {
                        greedy_actions[0].clone()
                    } else {
                        greedy_actions[rand::random::<usize>() % greedy_actions.len()].clone()
                    }
                } else {
                    random_action(&state)
                }
            };

            let state_action_value = *action_values
                .get(&state)
                .map_or(&0.0, |av| av.get(&action).unwrap_or(&0.0));

            // Take the action and determine the next state and the reward.
            let (maybe_new_state, reward) = next_state(&state, &action);

            // Update the state action value Q(S, A):
            //   Q(S, A) ← Q(S, A) + α∙[R + γ∙∑π(a|S)∙Q(S₊₁, a) - Q(S, A)],
            // where π(a|S) is the probability of taking action a under ε-greedy policy from Q.

            // If this is a final state, then formula above simplifies to:
            //   Q(S, A) ← Q(S, A) + α∙[R - Q(S, A)]
            if maybe_new_state.is_none() {
                let new_state_action_value =
                    state_action_value + alpha * (reward - state_action_value);
                action_values
                    .entry(state)
                    .or_default()
                    .insert(action, new_state_action_value);
                break;
            }

            let new_state = maybe_new_state.unwrap();

            // Compute the returns from state S₊₁.
            let returns = action_values
                .get(&new_state)
                .map(|av| {
                    assert!(!av.is_empty());

                    // If there is just a single action, then it's probability is 1.
                    if av.len() == 1 {
                        return *av.iter().nth(0).unwrap().1;
                    }

                    // Find the maximum action value.
                    let max_value = av
                        .iter()
                        .map(|(_, v)| v)
                        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));

                    // Find the number of actions with max action value (can be multiple!).
                    let greedy_actions_count = av
                        .iter()
                        .filter(|(_, v)| (*v - max_value).abs() < 1e-6)
                        .count();

                    assert!(greedy_actions_count > 0);

                    let others_probability = exploration_fraction / (av.len() as f64);
                    let greedy_probability = others_probability
                        + (1.0 - exploration_fraction) / (greedy_actions_count as f64);

                    av.iter()
                        .map(|(_, v)| match (*v - max_value).abs() < 1e-6 {
                            true => greedy_probability * v,
                            false => others_probability * v,
                        })
                        .sum()
                })
                .unwrap_or(0.0);

            // Now update Q(S, A).
            let new_state_action_value =
                state_action_value + alpha * (reward + discount * returns - state_action_value);
            action_values
                .entry(state)
                .or_default()
                .insert(action, new_state_action_value);

            state = new_state;
        }
    }

    action_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
    enum RandomWalkState {
        A,
        B,
        C,
        D,
        E,
    }

    #[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
    enum RandomWalkAction {
        Left,
        Right,
    }

    fn random_walk_start_state() -> RandomWalkState {
        RandomWalkState::C
    }

    fn random_walk_random_action(state: &RandomWalkState) -> RandomWalkAction {
        if rand::random::<f64>() < 0.5 {
            RandomWalkAction::Left
        } else {
            RandomWalkAction::Right
        }
    }

    fn random_walk_next_state(
        state: &RandomWalkState,
        action: &RandomWalkAction,
    ) -> (Option<RandomWalkState>, f64) {
        match action {
            RandomWalkAction::Left => match state {
                RandomWalkState::A => (None, 0.0),
                RandomWalkState::B => (Some(RandomWalkState::A), 0.0),
                RandomWalkState::C => (Some(RandomWalkState::B), 0.0),
                RandomWalkState::D => (Some(RandomWalkState::C), 0.0),
                RandomWalkState::E => (Some(RandomWalkState::D), 0.0),
            },
            RandomWalkAction::Right => match state {
                RandomWalkState::A => (Some(RandomWalkState::B), 0.0),
                RandomWalkState::B => (Some(RandomWalkState::C), 0.0),
                RandomWalkState::C => (Some(RandomWalkState::D), 0.0),
                RandomWalkState::D => (Some(RandomWalkState::E), 0.0),
                RandomWalkState::E => (None, 1.0),
            },
        }
    }

    #[test]
    fn expected_sarsa_random_walk_test() {
        use RandomWalkAction as A;
        use RandomWalkState as S;

        let discount = 1.0;
        let exploration_fraction = 1.0; // Make it a random policy.
        let alpha = 0.1;
        let iterations = 1000;
        let action_values = find_action_values_expected_sarsa(
            &random_walk_start_state,
            &random_walk_random_action,
            &random_walk_next_state,
            discount,
            exploration_fraction,
            alpha,
            iterations,
        );

        // Expected state values under random policy.
        let expected_state_values = [
            (S::A, 1.0 / 6.0),
            (S::B, 2.0 / 6.0),
            (S::C, 3.0 / 6.0),
            (S::D, 4.0 / 6.0),
            (S::E, 5.0 / 6.0),
        ];
        for (state, expected_state_value) in expected_state_values.iter() {
            let state_action_values = action_values.get(state).unwrap();
            let left_value = state_action_values.get(&A::Left).unwrap_or(&0.0);
            let right_value = state_action_values.get(&A::Right).unwrap_or(&0.0);
            let avg = (left_value + right_value) * 0.5;
            println!(
                "State {:?} L: {:.03}, R: {:.03} => {:.03} (expected {:.03})",
                state, left_value, right_value, avg, expected_state_value
            );
            assert!((avg - expected_state_value).abs() < 1e-3);
        }
    }
}
