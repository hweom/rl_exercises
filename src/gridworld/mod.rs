use std::collections::HashMap;

use prettytable::{Cell, Row, Table};

use crate::solver::*;

const UP: &'static str = "↑";
const DOWN: &'static str = "↓";
const LEFT: &'static str = "←";
const RIGHT: &'static str = "→";

pub fn new_state_id(row: i32, col: i32) -> StateId {
    format!("{}_{}", row, col)
}

pub fn new_grid_env(rows: i32, cols: i32) -> Env {
    let mut states = HashMap::new();
    for row in 0..rows {
        for col in 0..cols {
            let mut actions = HashMap::new();

            // Not a final state.
            if (row != 0 || col != 0) && (row != rows - 1 || col != cols - 1) {
                actions.insert(
                    UP.to_string(),
                    deterministic_action(new_state_id((row - 1).max(0), col), -1.0),
                );
                actions.insert(
                    DOWN.to_string(),
                    deterministic_action(new_state_id((row + 1).min(rows - 1), col), -1.0),
                );
                actions.insert(
                    LEFT.to_string(),
                    deterministic_action(new_state_id(row, (col - 1).max(0)), -1.0),
                );
                actions.insert(
                    RIGHT.to_string(),
                    deterministic_action(new_state_id(row, (col + 1).min(cols - 1)), -1.0),
                );
            }

            states.insert(new_state_id(row, col), State { actions: actions });
        }
    }
    Env { states: states }
}

pub fn new_grid_random_policy(env: &Env) -> Policy {
    let mut policy_states = HashMap::new();
    for (state_id, state) in &env.states {
        let total_actions = state.actions.len();
        let mut policy_state_actions = HashMap::new();
        for action_id in state.actions.keys() {
            policy_state_actions.insert(action_id.clone(), 1.0 / total_actions as f64);
        }
        policy_states.insert(
            state_id.clone(),
            PolicyState {
                actions: policy_state_actions,
            },
        );
    }

    Policy {
        states: policy_states,
    }
}

pub fn print_grid_state_values(state_values: &HashMap<StateId, f64>, rows: i32, cols: i32) {
    let mut table = Table::new();
    for r in 0..rows {
        let mut cells = Vec::new();
        for c in 0..cols {
            let state_id = new_state_id(r, c);
            let value = state_values.get(&state_id).unwrap_or(&0.0);
            cells.push(Cell::new(format!("{:.2}", value).as_ref()));
        }
        table.add_row(Row::new(cells));
    }
    table.printstd();
}

pub fn print_grid_policy(policy: &Policy, rows: i32, cols: i32) {
    let empty_policy_state = &PolicyState {
        actions: HashMap::new(),
    };
    let mut table = Table::new();
    for r in 0..rows {
        let mut cells = Vec::new();
        for c in 0..cols {
            let state_id = new_state_id(r, c);
            let policy_actions = &policy
                .states
                .get(&state_id)
                .unwrap_or(&empty_policy_state)
                .actions;

            let symbol = match policy_actions.len() {
                0 => " ",
                1 => policy_actions.iter().nth(0).unwrap().0,
                _ => "?",
            };

            cells.push(Cell::new(symbol));
        }
        table.add_row(Row::new(cells));
    }
    table.printstd();
}
