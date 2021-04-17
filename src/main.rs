mod gridworld;
mod solver;

use std::collections::HashMap;

use gridworld::*;
use solver::*;

fn main() {
    let rows = 10;
    let cols = 10;

    // Create environment.
    let env = new_grid_env(rows, cols);

    // Create policy.
    let policy = new_grid_random_policy(&env);

    let mut state_values = HashMap::new();
    for i in 0..10000 {
        let (new_state_values, delta) = iteration(&env, &policy, &state_values);
        state_values = new_state_values;
        if delta < 0.00001 {
            break;
        }
    }

    print_grid_state_values(&state_values, rows, cols);
    let greedy_policy = make_greedy_policy(&env, &state_values);
    print_grid_policy(&greedy_policy, rows, cols);
}
