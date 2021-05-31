pub mod explicit;
pub mod monte_carlo;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use rand::prelude::*;

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
