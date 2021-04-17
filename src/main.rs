use std::cmp;
use std::collections::HashMap;

use solver::*;

mod solver;

fn deterministic_action_result(dest_state: StateId, reward: f32) -> ActionResult {
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

fn new_state_id(row: i32, col: i32) -> StateId {
    format!("{}_{}", row, col)
}

fn main() {
    let up = "up".to_string();
    let down = "down".to_string();
    let left = "left".to_string();
    let right = "right".to_string();

    let rows = 10;
    let cols = 10;

    // Create environment.
    let mut states = HashMap::new();
    for row in 0..rows {
        for col in 0..cols {
            let mut actions = HashMap::new();

            // Not a final state.
            if (row != 0 || col != 0) && (row != rows - 1 || col != cols - 1) {
                actions.insert(
                    up.clone(),
                    deterministic_action_result(new_state_id((row - 1).max(0), col), -1.0),
                );
                actions.insert(
                    down.clone(),
                    deterministic_action_result(new_state_id((row + 1).min(rows - 1), col), -1.0),
                );
                actions.insert(
                    left.clone(),
                    deterministic_action_result(new_state_id(row, (col - 1).max(0)), -1.0),
                );
                actions.insert(
                    right.clone(),
                    deterministic_action_result(new_state_id(row, (col + 1).min(cols - 1)), -1.0),
                );
            }

            states.insert(new_state_id(row, col), State { actions: actions });
        }
    }
    let env = Env { states: states };

    // Create policy.
    let mut policy_states = HashMap::new();
    for (state_id, state) in &env.states {
        let total_actions = state.actions.len();
        let mut policy_state_actions = HashMap::new();
        for action_id in state.actions.keys() {
            policy_state_actions.insert(action_id.clone(), 1.0 / total_actions as f32);
        }
        policy_states.insert(
            state_id.clone(),
            PolicyState {
                actions: policy_state_actions,
            },
        );
    }
    let policy = Policy {
        states: policy_states,
    };

    let mut state_values = HashMap::new();
    for i in 0..10000 {
        let (new_state_values, delta) = iteration(&env, &policy, &state_values);
        state_values = new_state_values;
        println!("{}: {}", i, delta);
        if delta < 0.01 {
            break;
        }
    }

    for row in 0..rows {
        for col in 0..cols {
            let state_id = new_state_id(row, col);
            let state_value = state_values.get(&state_id).unwrap_or(&0.0);
            print!("{:+03.2} ", state_value);
        }
        println!();
    }
}
