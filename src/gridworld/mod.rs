use std::collections::HashMap;
use std::fmt;

use prettytable::{Cell, Row, Table};

use crate::solver::*;

const UP: &'static str = "↑";
const DOWN: &'static str = "↓";
const LEFT: &'static str = "←";
const RIGHT: &'static str = "→";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    row: i32,
    col: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl State {
    pub fn new(r: i32, c: i32) -> State {
        State { row: r, col: c }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}:{})", self.row, self.col)
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Action::Up => UP,
                Action::Down => DOWN,
                Action::Left => LEFT,
                Action::Right => RIGHT,
            }
        )
    }
}

pub fn new_grid_env(rows: i32, cols: i32) -> Env<State, Action> {
    let mut states = HashMap::new();
    for row in 0..rows {
        for col in 0..cols {
            let mut actions = HashMap::new();

            // Not a final state.
            if (row != 0 || col != 0) && (row != rows - 1 || col != cols - 1) {
                actions.insert(
                    Action::Up,
                    deterministic_action(State::new((row - 1).max(0), col), -1.0),
                );
                actions.insert(
                    Action::Down,
                    deterministic_action(State::new((row + 1).min(rows - 1), col), -1.0),
                );
                actions.insert(
                    Action::Left,
                    deterministic_action(State::new(row, (col - 1).max(0)), -1.0),
                );
                actions.insert(
                    Action::Right,
                    deterministic_action(State::new(row, (col + 1).min(cols - 1)), -1.0),
                );
            }

            states.insert(State::new(row, col), StateActions { actions: actions });
        }
    }
    Env { states: states }
}

pub fn new_grid_random_policy(env: &Env<State, Action>) -> Policy<State, Action> {
    let mut policy_states = HashMap::new();
    for (state, state_actions) in &env.states {
        let total_actions = state_actions.actions.len();
        let mut policy_state_actions = HashMap::new();
        for action in state_actions.actions.keys() {
            policy_state_actions.insert(*action, 1.0 / total_actions as f64);
        }
        policy_states.insert(
            *state,
            PolicyState {
                actions: policy_state_actions,
            },
        );
    }

    Policy {
        states: policy_states,
    }
}

pub fn print_grid_state_values(state_values: &HashMap<State, f64>, rows: i32, cols: i32) {
    let mut table = Table::new();
    for r in 0..rows {
        let mut cells = Vec::new();
        for c in 0..cols {
            let state_id = State::new(r, c);
            let value = state_values.get(&state_id).unwrap_or(&0.0);
            cells.push(Cell::new(format!("{:.2}", value).as_ref()));
        }
        table.add_row(Row::new(cells));
    }
    table.printstd();
}

pub fn print_grid_policy(policy: &Policy<State, Action>, rows: i32, cols: i32) {
    let empty_policy_state = &PolicyState {
        actions: HashMap::new(),
    };
    let mut table = Table::new();
    for r in 0..rows {
        let mut cells = Vec::new();
        for c in 0..cols {
            let state_id = State::new(r, c);
            let policy_actions = &policy
                .states
                .get(&state_id)
                .unwrap_or(&empty_policy_state)
                .actions;

            match policy_actions.len() {
                0 => cells.push(Cell::new(" ")),
                1 => cells.push(Cell::new(
                    format!("{}", policy_actions.iter().nth(0).unwrap().0).as_str(),
                )),
                _ => cells.push(Cell::new("?")),
            };
        }
        table.add_row(Row::new(cells));
    }
    table.printstd();
}
