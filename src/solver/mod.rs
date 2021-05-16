use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use rand::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct ActionDestination {
    pub probability: f64,
    pub reward: f64,
}

#[derive(Debug, Default, Clone)]
pub struct ActionResult<S: Eq + Hash> {
    // Possible destination states with associated probabilities and rewards.
    // All destination probabilities must sum to 1.
    pub dest_states: HashMap<S, ActionDestination>,
}

#[derive(Debug, Default, Clone)]
pub struct StateActions<S: Eq + Hash, A: Eq + Hash> {
    // Possible actions in this state.
    // Empty if this is a final state.
    pub actions: HashMap<A, ActionResult<S>>,
}

#[derive(Debug, Default, Clone)]
pub struct Env<S: Eq + Hash, A: Eq + Hash> {
    pub states: HashMap<S, StateActions<S, A>>,
}

#[derive(Debug, Default, Clone)]
pub struct PolicyState<A: Eq + Hash> {
    // Possible actions and their probabilities.
    // All probabilities must sum to 1.
    pub actions: HashMap<A, f64>,
}

#[derive(Debug, Default, Clone)]
pub struct Policy<S: Eq + Hash, A: Eq + Hash> {
    pub states: HashMap<S, PolicyState<A>>,
}

// Returns the action value given the action results and state value function.
fn get_action_value<S: Eq + Hash>(
    action: &ActionResult<S>,
    state_values: &HashMap<S, f64>,
    discount: f64,
) -> f64 {
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
    K: Copy + Ord + Hash + Eq,
    F: FnMut(&V) -> f64,
{
    let total_probablity: f64 = map.iter().map(|(k, v)| f(v)).sum();

    let mut keys: Vec<&K> = map.keys().collect();
    keys.sort();

    let mut remaining_probability = rand::random::<f64>() * total_probablity;
    for k in keys.iter() {
        let probability = f(map.get(k).unwrap());
        if remaining_probability <= probability {
            return **k;
        }

        remaining_probability = remaining_probability - probability;
    }

    // Can't be here.
    panic!();
}

pub fn deterministic_action<S: Eq + Hash>(dest_state: S, reward: f64) -> ActionResult<S> {
    let mut dest_states = HashMap::new();
    dest_states.insert(
        dest_state,
        ActionDestination {
            probability: 1.0,
            reward: reward,
        },
    );
    ActionResult {
        dest_states: dest_states,
    }
}

// Performs a single iteration to determine the next state-value function.
// Returns new state-value function and a maximum change in state-values.
pub fn evaluate_policy<S: Copy + Eq + Hash + Debug, A: Eq + Hash>(
    env: &Env<S, A>,
    policy: &Policy<S, A>,
    prev_state_values: &HashMap<S, f64>,
    discount: f64,
) -> (HashMap<S, f64>, f64) {
    let mut new_state_values = HashMap::new();
    let mut max_delta: f64 = 0.0;

    for (state, state_actions) in env.states.iter() {
        if state_actions.actions.is_empty() {
            continue;
        }

        let state_policy = policy
            .states
            .get(state)
            .expect(&format!("No policy for state {:?}", state));

        let mut state_value = 0.0;
        for (action, action_result) in state_actions.actions.iter() {
            let action_value = get_action_value(action_result, prev_state_values, discount);
            let action_prob = state_policy.actions.get(action).unwrap_or(&0.0);
            state_value += action_value * action_prob;
        }

        let prev_state_value = prev_state_values.get(state).unwrap_or(&0.0);
        max_delta = max_delta.max((prev_state_value - state_value).abs());
        new_state_values.insert(*state, state_value);
    }

    (new_state_values, max_delta)
}

// Performs a single state value function iteration.
// Returns new state-value function and a maximum change in state-values.
pub fn iterate_state_value<S: Copy + Eq + Hash, A: Eq + Hash>(
    env: &Env<S, A>,
    prev_state_values: &HashMap<S, f64>,
    discount: f64,
) -> (HashMap<S, f64>, f64) {
    let mut new_state_values = HashMap::new();
    let mut max_delta: f64 = 0.0;

    for (state, state_actions) in env.states.iter() {
        if state_actions.actions.is_empty() {
            continue;
        }

        let best_action_value = state_actions
            .actions
            .iter()
            .map(|(_, action_result)| get_action_value(action_result, prev_state_values, discount))
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        new_state_values.insert(*state, best_action_value);

        let prev_state_value = prev_state_values.get(state).unwrap_or(&0.0);
        max_delta = max_delta.max(best_action_value - prev_state_value);
    }

    (new_state_values, max_delta)
}

pub fn make_uniform_policy<S: Copy + Eq + Hash, A: Copy + Eq + Hash>(
    env: &Env<S, A>,
) -> Policy<S, A> {
    let mut policy_states = HashMap::new();

    for (state, state_actions) in &env.states {
        if state_actions.actions.is_empty() {
            continue;
        }

        let action_probability = 1.0 / state_actions.actions.len() as f64;
        let policy_state = PolicyState {
            actions: state_actions
                .actions
                .keys()
                .map(|action| (*action, action_probability))
                .collect(),
        };
        policy_states.insert(*state, policy_state);
    }

    Policy {
        states: policy_states,
    }
}

pub fn make_greedy_policy<S: Copy + Eq + Hash, A: Copy + Eq + Hash>(
    env: &Env<S, A>,
    state_values: &HashMap<S, f64>,
    discount: f64,
) -> Policy<S, A> {
    let mut policy_states = HashMap::new();

    for (state, state_actions) in &env.states {
        if state_actions.actions.is_empty() {
            continue;
        }

        // Determine maximum possible reward.
        let max_action_reward = state_actions
            .actions
            .iter()
            .map(|(_, action_actions)| get_action_value(action_actions, state_values, discount))
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Find action IDs that yield the maximum reward.
        let actions: Vec<&A> = state_actions
            .actions
            .iter()
            .filter(|(_, action_result)| {
                // Don't require exact equality to forgive rounding errors.
                (get_action_value(action_result, state_values, discount) - max_action_reward).abs()
                    < 1e-6
            })
            .map(|(action, _)| action)
            .collect();

        assert!(!actions.is_empty());

        let action_probability = 1.0 / actions.len() as f64;

        // Create a policy that takes either of the max reward action with equal probability.
        policy_states.insert(
            *state,
            PolicyState {
                actions: actions
                    .iter()
                    .map(|action| (**action, action_probability))
                    .collect(),
            },
        );
    }

    Policy {
        states: policy_states,
    }
}

pub fn run_simulation<S: Copy + Eq + Hash + Debug + Ord, A: Copy + Eq + Hash + Ord>(
    env: &Env<S, A>,
    policy: &Policy<S, A>,
    start_state: S,
    max_steps: u32,
) -> f64 {
    let mut state = start_state;
    let mut total_reward = 0.0;
    for _ in 0..max_steps {
        let state_actions = env
            .states
            .get(&state)
            .expect(format!("State {:?} not found", state).as_str());

        // Final state.
        if state_actions.actions.is_empty() {
            break;
        }

        // Choose action stochastically.
        let policy_state = policy.states.get(&state).unwrap();
        let action = choose_random_key(&policy_state.actions, |v| *v);
        let action_results = state_actions.actions.get(&action).unwrap();

        // Choose end state stochastically.
        let target_state = choose_random_key(&action_results.dest_states, |v| v.probability);

        total_reward = total_reward
            + action_results
                .dest_states
                .get(&target_state)
                .unwrap()
                .reward;
        state = target_state;
    }

    total_reward
}
