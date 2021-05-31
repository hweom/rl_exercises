use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::once;

use crate::solver::explicit::{Policy, PolicyState};
use crate::solver::*;

#[derive(Clone, Debug, Default)]
struct ValueEstimate {
    avg: f64,
    count: u32,
}

#[derive(Clone, Debug)]
struct StateActionEstimate<A: Hash + Eq> {
    actions: HashMap<A, ValueEstimate>,
}

impl ValueEstimate {
    fn update(&mut self, value: f64) {
        self.avg = (self.avg * (self.count as f64) + value) / (self.count + 1) as f64;
        self.count += 1
    }
}

impl<A: Hash + Eq> Default for StateActionEstimate<A> {
    fn default() -> StateActionEstimate<A> {
        StateActionEstimate {
            actions: HashMap::new(),
        }
    }
}

pub fn policy_from_explicit<S, A>(explicit_policy: Policy<S, A>) -> Box<dyn Fn(&S) -> A>
where
    S: Eq + Hash + 'static,
    A: Eq + Hash + Clone + Ord + 'static,
{
    Box::new(move |s| {
        // Choose action stochastically.
        let policy_state = explicit_policy.states.get(s).unwrap();
        choose_random_key(&policy_state.actions, |v| *v)
    })
}

pub fn evaluate_policy<S, A, StartState, Policy, NextState>(
    start_state: &StartState,
    policy: &Policy,
    next_state: &NextState,
    discount: f64,
    iterations: u64,
) -> HashMap<S, f64>
where
    S: Eq + Hash + Debug + Clone,
    A: Eq + Hash,
    StartState: Fn() -> S,
    Policy: Fn(&S) -> A,
    NextState: Fn(&S, &A) -> (Option<S>, f64),
{
    let mut state_values = HashMap::new();

    for _ in 0..iterations {
        // Generate a single episode.
        let mut state = start_state();
        let mut episode = Vec::new();
        loop {
            let action = policy(&state);
            let (new_state, reward) = next_state(&state, &action);
            episode.push((state, action, reward));
            if new_state.is_none() {
                break;
            }
            state = new_state.unwrap();
        }

        // Update state values from this episode.
        let mut updated_states = HashSet::new();
        let mut returns = 0.0;
        while !episode.is_empty() {
            let (state, _action, reward) = episode.pop().unwrap();
            returns = returns * discount + reward;
            if updated_states.insert(state.clone()) {
                state_values
                    .entry(state)
                    .or_insert_with(|| ValueEstimate::default())
                    .update(returns);
            }
        }
    }

    return state_values
        .into_iter()
        .map(|(state, estimation)| (state, estimation.avg))
        .collect();
}

pub fn find_policy<S, A, StartState, RandomAction, NextState>(
    start_state: &StartState,
    random_action: &RandomAction,
    next_state: &NextState,
    discount: f64,
    exploration_fraction: f64,
    iterations: u64,
) -> Policy<S, A>
where
    S: Eq + Hash + Debug + Clone,
    A: Eq + Hash + Debug + Clone,
    StartState: Fn() -> S,
    RandomAction: Fn(&S) -> A,
    NextState: Fn(&S, &A) -> (Option<S>, f64),
{
    let mut action_values: HashMap<S, StateActionEstimate<A>> = HashMap::new();

    for _ in 0..iterations {
        // Generate a single episode.
        let mut state = start_state();
        let mut episode: Vec<(S, A, f64)> = Vec::new();
        loop {
            // Determine the next action.
            let action = match action_values.get(&state) {
                // This state has already been visited -- choose best known action with
                // (1 - exploration_fraction) probability or othewise choose random one.
                Some(state_action_values) => {
                    if rand::random::<f64>() <= exploration_fraction {
                        random_action(&state)
                    } else {
                        state_action_values
                            .actions
                            .iter()
                            .max_by(|(_, e1), (_, e2)| e1.avg.partial_cmp(&e2.avg).unwrap())
                            .unwrap()
                            .0
                            .clone()
                    }
                }
                // No actions explored for this state -- choose action at random.
                None => random_action(&state),
            };

            let (new_state, reward) = next_state(&state, &action);
            episode.push((state, action, reward));
            if new_state.is_none() {
                break;
            }
            state = new_state.unwrap();
        }

        // Update state values from this episode.
        let mut returns = 0.0;
        let mut observed_state_actions = HashMap::new();
        while !episode.is_empty() {
            let (state, action, reward) = episode.pop().unwrap();
            returns = returns * discount + reward;
            observed_state_actions.insert((state, action), returns);
        }
        for ((state, action), returns) in observed_state_actions {
            action_values
                .entry(state)
                .or_insert_with(|| StateActionEstimate::default())
                .actions
                .entry(action)
                .or_insert_with(|| ValueEstimate::default())
                .update(returns);
        }
    }
    Policy {
        states: action_values
            .into_iter()
            .map(|(state, actions)| {
                let best_action = actions
                    .actions
                    .into_iter()
                    .max_by(|(_, e1), (_, e2)| e1.avg.partial_cmp(&e2.avg).unwrap())
                    .unwrap()
                    .0;
                let policy_state_actions: HashMap<A, f64> = once((best_action, 1.0)).collect();
                (
                    state,
                    PolicyState {
                        actions: policy_state_actions,
                    },
                )
            })
            .collect(),
    }
}

pub fn run_simulation<S, A, StartState, Policy, NextState>(
    start_state: &StartState,
    policy: &Policy,
    next_state: &NextState,
) -> f64
where
    S: Eq + Hash + Debug + Clone,
    A: Eq + Hash + Debug + Clone + Ord,
    StartState: Fn() -> S,
    Policy: Fn(&S) -> A,
    NextState: Fn(&S, &A) -> (Option<S>, f64),
{
    let mut returns = 0.0;
    let mut state = start_state();
    loop {
        let action = policy(&state);

        // Get to the next state and collect reward.
        let (new_state, reward) = next_state(&state, &action);
        returns += reward;
        if new_state.is_none() {
            break;
        }
        state = new_state.unwrap();
    }
    returns
}
