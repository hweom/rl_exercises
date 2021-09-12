use nalgebra::DVector;

use crate::solver::*;

fn soft_greedy_action<S, A, I, StateActionFeatures>(
    actions: &Vec<A>,
    w: &DVector<f64>,
    state_action_features: &StateActionFeatures,
    state: &S,
    action_indices: I,
    exploration_fraction: f64,
) -> usize
where
    I: Iterator<Item = usize>,
    StateActionFeatures: Fn(&S, &A) -> Vec<f64>,
{
    // If we pass the exploration check, choose the action at random.
    if rand::random::<f64>() <= exploration_fraction {
        let all_action_indices: Vec<usize> = action_indices.collect();
        return all_action_indices[rand::random::<usize>() % all_action_indices.len()].clone();
    }

    // Go over the actions and find the "best" ones (ones having maximum value).
    let mut best_action_indices = Vec::new();
    let mut best_value = f64::NEG_INFINITY;
    for a in action_indices {
        let features = DVector::from_vec(state_action_features(state, &actions[a]));
        let value = w.dot(&features);
        if value > best_value {
            best_action_indices.clear();
            best_action_indices.push(a);
            best_value = value;
        }
    }

    // Now choose at random between all "best" actions (trivial if there is only one).
    if best_action_indices.len() == 1 {
        best_action_indices[0]
    } else {
        return best_action_indices[rand::random::<usize>() % best_action_indices.len()].clone();
    }
}

pub fn find_action_values_episodic_semi_gradient_sarsa<
    S,
    A: Eq + Hash + Clone,
    StartState,
    StateActionFeatures,
    IsActionPossible,
    NextState,
>(
    actions: &Vec<A>,
    start_state: &StartState,
    state_action_features: &StateActionFeatures,
    is_action_possible: &IsActionPossible,
    next_state: &NextState,
    discount: f64,
    exploration_fraction: f64,
    alpha: f64,
    iterations: usize,
) -> DVector<f64>
where
    StartState: Fn() -> S,
    StateActionFeatures: Fn(&S, &A) -> Vec<f64>,
    IsActionPossible: Fn(&S, &A) -> bool,
    NextState: Fn(&S, &A) -> (Option<S>, f64),
{
    // Determine state features count by creating a dummy start state.
    let state_feature_count = {
        let start_state = start_state();
        let action = actions
            .iter()
            .find(|a| is_action_possible(&start_state, a))
            .unwrap();
        state_action_features(&start_state, action).len()
    };

    // Store weights independently for each action (using action index as a key).
    let mut w = DVector::repeat(state_feature_count, 0.0);

    for _ in 0..iterations {
        // Generate a single episode.

        // Generate the starting state and action from it.
        let mut state = start_state();
        let mut action_index = soft_greedy_action(
            actions,
            &w,
            state_action_features,
            &state,
            (0..actions.len()).filter(|i| is_action_possible(&state, &actions[*i])),
            exploration_fraction,
        );
        let mut features = DVector::from_vec(state_action_features(&state, &actions[action_index]));

        // Go to the next state until a final state is reached.
        loop {
            // Take the action and determine the next state and the reward.
            let (maybe_next_state, reward) = next_state(&state, &actions[action_index]);

            // Update the state action value approximation q̂(S, A, w):
            //   w ← w + α∙[R + γ∙q̂(S₊₁, A₊₁, w) - q̂(S, A, w)]∙∇q̂(S, A, w).
            // For linear state approximation q̂(S, A, w) = w∙x(S, A),
            // the gradient is just:
            //   ∇q̂(S, A, w) = x(S, A).

            // Compute previous action value q̂(S, A, w).
            let prev_action_value = w.dot(&features);

            // If this is a final state, then formula above simplifies to:
            //   w ← w + α∙[R - q̂(S, A, w)]∙∇q̂(S, A, w),
            if maybe_next_state.is_none() {
                w = w + alpha * (reward - prev_action_value) * features;
                break;
            }

            let next_state = maybe_next_state.unwrap();

            // Determine next action to compute the expected returns.
            let next_action_index = soft_greedy_action(
                actions,
                &w,
                state_action_features,
                &next_state,
                (0..actions.len()).filter(|i| is_action_possible(&next_state, &actions[*i])),
                exploration_fraction,
            );
            let next_features = DVector::from_vec(state_action_features(&next_state, &actions[next_action_index]));

            // Compute expected returns.
            let expected_returns = reward + discount * w.dot(&next_features);

            // Update the approximation weights.
            w = w + alpha * (expected_returns - prev_action_value) * features;

            state = next_state;
            features = next_features;
            action_index = next_action_index;
        }
    }

    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Eq, Hash, Clone)]
    enum RandomWalkAction {
        Left,
        Right,
    }

    #[test]
    fn episodic_semi_gradient_sarsa_random_walk_test() {
        use RandomWalkAction as A;

        let start_state = || rand::random::<usize>() % 100;
        let state_action_features = |s: &usize, a: &A| vec![*s as f64];
        let is_action_possible = |s: &usize, a: &A| match a {
            A::Left => *s > 0,
            A::Right => true,
        };
        let next_state = |s: &usize, a: &A| match a {
            A::Left => {
                assert!(*s > 0);
                (Some(*s - 1), -1.0)
            }
            A::Right => {
                if *s >= 99 {
                    (None, 0.0)
                } else {
                    (Some(*s + 1), -1.0)
                }
            }
        };

        let discount = 1.0;
        let exploration_fraction = 0.1;
        let alpha = 0.1;
        let iterations = 1;
        let value = find_action_values_episodic_semi_gradient_sarsa(
            &vec![A::Left, A::Right],
            &start_state,
            &state_action_features,
            &is_action_possible,
            &next_state,
            discount,
            exploration_fraction,
            alpha,
            iterations,
        );
    }
}
