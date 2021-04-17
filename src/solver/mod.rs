use std::collections::HashMap;

pub use String as StateId;
pub use String as ActionId;

pub struct ActionDestination {
    pub probability: f32,
    pub reward: f32,
}

pub struct Action {
    // Possible destination states with associated probabilities and rewards.
    // All destination probabilities must sum to 1.
    pub dest_states: HashMap<StateId, ActionDestination>,
}

pub struct State {
    // Possible actions in this state.
    // Empty if this is a final state.
    pub actions: HashMap<ActionId, Action>,
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

// Returns the action value given the action and state value function.
fn get_action_value(action: &Action, state_values: &HashMap<StateId, f32>) -> f32 {
    action
        .dest_states
        .iter()
        .map(|(state_id, dest)| {
            dest.probability * (dest.reward + state_values.get(state_id).unwrap_or(&0.0))
        })
        .sum()
}

pub fn deterministic_action(dest_state: StateId, reward: f32) -> Action {
    let mut dest_states = HashMap::new();
    dest_states.insert(
        dest_state,
        ActionDestination {
            probability: 1.0,
            reward: reward,
        },
    );
    Action {
        dest_states: dest_states,
    }
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
        for (action_id, action) in state.actions.iter() {
            let action_value = get_action_value(action, prev_state_values);
            let action_prob = state_policy.actions.get(action_id).unwrap_or(&0.0);
            state_value += action_value * action_prob;
        }

        let prev_state_value = prev_state_values.get(state_id).unwrap_or(&0.0);
        max_delta = max_delta.max((prev_state_value - state_value).abs());
        new_state_values.insert(state_id.clone(), state_value);
    }

    (new_state_values, max_delta)
}

pub fn make_greedy_policy(env: &Env, state_values: &HashMap<StateId, f32>) -> Policy {
    let mut policy_states = HashMap::new();

    for (state_id, state) in &env.states {
        if state.actions.is_empty() {
            continue;
        }

        // Determine maximum possible reward.
        let max_action_reward = state
            .actions
            .iter()
            .map(|(_, action)| get_action_value(action, state_values))
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // Find action IDs that yield the maximum reward.
        let action_ids: Vec<&ActionId> = state
            .actions
            .iter()
            .filter(|(_, action)| get_action_value(action, state_values) == max_action_reward)
            .map(|(action_id, _)| action_id)
            .collect();

        assert!(!action_ids.is_empty());

        let action_probability = 1.0 / action_ids.len() as f32;

        // Create a policy that takes either of the max reward action with equal probability.
        policy_states.insert(
            state_id.clone(),
            PolicyState {
                actions: action_ids
                    .iter()
                    .map(|action_id| (action_id.to_string(), action_probability))
                    .collect(),
            },
        );
    }

    Policy {
        states: policy_states,
    }
}
