use std::collections::HashMap;
use std::hash::Hash;

use rand::prelude::*;

pub use String as StateId;
pub use String as ActionId;

#[derive(Debug, Default, Clone)]
pub struct ActionDestination {
    pub probability: f64,
    pub reward: f64,
}

#[derive(Debug, Default, Clone)]
pub struct Action {
    // Possible destination states with associated probabilities and rewards.
    // All destination probabilities must sum to 1.
    pub dest_states: HashMap<StateId, ActionDestination>,
}

#[derive(Debug, Default, Clone)]
pub struct State {
    // Possible actions in this state.
    // Empty if this is a final state.
    pub actions: HashMap<ActionId, Action>,
}

#[derive(Debug, Default, Clone)]
pub struct Env {
    pub states: HashMap<StateId, State>,
}

#[derive(Debug, Default, Clone)]
pub struct PolicyState {
    // Possible actions and their probabilities.
    // All probabilities must sum to 1.
    pub actions: HashMap<ActionId, f64>,
}

#[derive(Debug, Default, Clone)]
pub struct Policy {
    pub states: HashMap<StateId, PolicyState>,
}

// Returns the action value given the action and state value function.
fn get_action_value(action: &Action, state_values: &HashMap<StateId, f64>, discount: f64) -> f64 {
    action
        .dest_states
        .iter()
        .map(|(state_id, dest)| {
            dest.probability * (dest.reward + discount * state_values.get(state_id).unwrap_or(&0.0))
        })
        .sum()
}

fn choose_random_key<K, V, F>(map: &HashMap<K, V>, mut f: F) -> K
where
    K: Clone + Ord + Hash + Eq,
    F: FnMut(&V) -> f64,
{
    let total_probablity: f64 = map.iter().map(|(k, v)| f(v)).sum();

    let mut keys: Vec<&K> = map.keys().collect();
    keys.sort();

    let mut remaining_probability = rand::random::<f64>() * total_probablity;
    for k in keys.iter() {
        let probability = f(map.get(k).unwrap());
        if remaining_probability <= probability {
            return (*k).clone();
        }

        remaining_probability = remaining_probability - probability;
    }

    // Can't be here.
    panic!();
}

pub fn deterministic_action(dest_state: StateId, reward: f64) -> Action {
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
pub fn evaluate_policy(
    env: &Env,
    policy: &Policy,
    prev_state_values: &HashMap<StateId, f64>,
    discount: f64,
) -> (HashMap<StateId, f64>, f64) {
    let mut new_state_values = HashMap::new();
    let mut max_delta: f64 = 0.0;

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
            let action_value = get_action_value(action, prev_state_values, discount);
            let action_prob = state_policy.actions.get(action_id).unwrap_or(&0.0);
            state_value += action_value * action_prob;
        }

        let prev_state_value = prev_state_values.get(state_id).unwrap_or(&0.0);
        max_delta = max_delta.max((prev_state_value - state_value).abs());
        new_state_values.insert(state_id.clone(), state_value);
    }

    (new_state_values, max_delta)
}

// Performs a single state value function iteration.
// Returns new state-value function and a maximum change in state-values.
pub fn iterate_state_value(
    env: &Env,
    prev_state_values: &HashMap<StateId, f64>,
    discount: f64,
) -> (HashMap<StateId, f64>, f64) {
    let mut new_state_values = HashMap::new();
    let mut max_delta: f64 = 0.0;

    for (state_id, state) in env.states.iter() {
        if state.actions.is_empty() {
            continue;
        }

        let best_action_value = state
            .actions
            .iter()
            .map(|(_action_id, action)| get_action_value(action, prev_state_values, discount))
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        new_state_values.insert(state_id.clone(), best_action_value);

        let prev_state_value = prev_state_values.get(state_id).unwrap_or(&0.0);
        max_delta = max_delta.max(best_action_value - prev_state_value);
    }

    (new_state_values, max_delta)
}

pub fn make_uniform_policy(env: &Env) -> Policy {
    let mut policy_states = HashMap::new();

    for (state_id, state) in &env.states {
        if state.actions.is_empty() {
            continue;
        }

        let action_probability = 1.0 / state.actions.len() as f64;
        let policy_state = PolicyState {
            actions: state
                .actions
                .keys()
                .map(|action_id| ((*action_id).clone(), action_probability))
                .collect(),
        };
        policy_states.insert(state_id.clone(), policy_state);
    }

    Policy {
        states: policy_states,
    }
}

pub fn make_greedy_policy(
    env: &Env,
    state_values: &HashMap<StateId, f64>,
    discount: f64,
) -> Policy {
    let mut policy_states = HashMap::new();

    for (state_id, state) in &env.states {
        if state.actions.is_empty() {
            continue;
        }

        // Determine maximum possible reward.
        let max_action_reward = state
            .actions
            .iter()
            .map(|(_, action)| get_action_value(action, state_values, discount))
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Find action IDs that yield the maximum reward.
        let action_ids: Vec<&ActionId> = state
            .actions
            .iter()
            .filter(|(_, action)| {
                // Don't require exact equality to forgive rounding errors.
                (get_action_value(action, state_values, discount) - max_action_reward).abs() < 1e-6
            })
            .map(|(action_id, _)| action_id)
            .collect();

        assert!(!action_ids.is_empty());

        let action_probability = 1.0 / action_ids.len() as f64;

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

pub fn run_simulation(env: &Env, policy: &Policy, start_state_id: &StateId, max_steps: u32) -> f64 {
    let mut state_id = start_state_id.clone();
    let mut total_reward = 0.0;
    for _ in 0..max_steps {
        let state = env
            .states
            .get(&state_id)
            .expect(format!("State {} not found", state_id).as_str());

        // Final state.
        if state.actions.is_empty() {
            break;
        }

        // Choose action stochastically.
        let policy_state = policy.states.get(&state_id).unwrap();
        let action_id = choose_random_key(&policy_state.actions, |v| *v);
        let action = state.actions.get(&action_id).unwrap();

        // Choose end state stochastically.
        let target_state_id = choose_random_key(&action.dest_states, |v| v.probability);

        total_reward = total_reward + action.dest_states.get(&target_state_id).unwrap().reward;
        state_id = target_state_id;
    }

    total_reward
}
