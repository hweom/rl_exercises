use std::collections::HashMap;

pub use String as StateId;
pub use String as ActionId;

pub struct ActionDestination {
    pub probability: f32,
    pub reward: f32,
}

pub struct ActionResult {
    // Possible destination states with associated probabilities and rewards.
    // All destination probabilities must sum to 1.
    pub dest_states: HashMap<StateId, ActionDestination>,
}

pub struct State {
    // Possible actions in this state.
    // Empty if this is a final state.
    pub actions: HashMap<ActionId, ActionResult>,
}

pub struct Env {
    pub states: HashMap<StateId, State>,
}

pub struct PolicyState {
    // Possible actions and their probabilities.
    // All probabilities must sum to 1.
    pub actions: HashMap<ActionId, f32>,
}

pub struct Policy {
    pub states: HashMap<StateId, PolicyState>,
}

// Performs a single iteration to determine the next state-value function.
// Returns new state-value function and a maximum change in state-values.
pub fn iteration(
    env: &Env,
    policy: &Policy,
    prev_state_values: &HashMap<StateId, f32>,
) -> (HashMap<StateId, f32>, f32) {
    let mut new_state_values = HashMap::new();
    let mut max_delta: f32 = 0.0;

    for (state_id, state) in env.states.iter() {
        if state.actions.is_empty() {
            continue;
        }

        let state_policy = policy
            .states
            .get(state_id)
            .expect(&format!("No policy for state {}", state_id));

        let mut state_value = 0.0;
        for (action_id, action_result) in state.actions.iter() {
            //            print!("Action {}_{} -> ", state_id, action_id);
            let action_value: f32 = action_result
                .dest_states
                .iter()
                .map(|(state_id, dest)| {
                    dest.probability
                        * (dest.reward + prev_state_values.get(state_id).unwrap_or(&0.0))
                })
                .sum();

            let action_prob = state_policy.actions.get(action_id).unwrap_or(&0.0);
            state_value += action_value * action_prob;
        }

        let prev_state_value = prev_state_values.get(state_id).unwrap_or(&0.0);
        max_delta = max_delta.max((prev_state_value - state_value).abs());
        new_state_values.insert(state_id.clone(), state_value);
    }

    (new_state_values, max_delta)
}
