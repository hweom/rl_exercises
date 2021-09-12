pub mod approximate;
pub mod explicit;
pub mod monte_carlo;
pub mod td;
pub mod tile;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::once;

use rand::prelude::*;

#[derive(Clone, Debug, Default)]
struct ValueEstimate {
    avg: f64,
    count: u32,
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

impl ValueEstimate {
    fn update(&mut self, value: f64) {
        self.avg = (self.avg * (self.count as f64) + value) / (self.count + 1) as f64;
        self.count += 1
    }
}

impl From<ValueEstimate> for f64 {
    fn from(estimate: ValueEstimate) -> f64 {
        estimate.avg
    }
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

fn policy_from_state_action_values<S, A, V>(
    action_values: HashMap<S, HashMap<A, V>>,
) -> Policy<S, A>
where
    S: Eq + Hash,
    A: Eq + Hash,
    V: Clone + Into<f64>,
{
    Policy {
        states: action_values
            .into_iter()
            .map(|(state, actions)| {
                let best_action = actions
                    .into_iter()
                    .map(|(a, v)| {
                        let f: f64 = v.into();
                        (a, f)
                    })
                    .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
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
