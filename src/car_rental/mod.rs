use std::collections::HashMap;

use factorial::Factorial;
use prettytable::{Cell, Row, Table};

use crate::solver::*;

const MAX_MOVES: i32 = 5;
const RENT_REWARD: f64 = 10.0;
const TRANSFER_PRICE: f64 = 2.0;
const RENTALS_LAMBDA1: f64 = 3.0;
const RENTALS_LAMBDA2: f64 = 4.0;
const RETURNS_LAMBDA1: f64 = 3.0;
const RETURNS_LAMBDA2: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    // Number of cars on location 1 and 2.
    l1: i32,
    l2: i32,
}

impl State {
    pub fn new(l1: i32, l2: i32) -> State {
        State { l1: l1, l2: l2 }
    }
}

fn poisson_prob(lambda: f64, n: i32) -> f64 {
    assert!(n >= 0);
    (-lambda).exp() * lambda.powi(n) / ((n as u128).factorial() as f64)
}

fn poisson_tail(lambda: f64, n: i32) -> f64 {
    1.0 - (0..n).map(|i| poisson_prob(lambda, i)).sum::<f64>()
}

fn new_action_result(
    max_cars: i32,
    l1_day: i32,
    l2_day: i32,
    transfer: i32,
) -> ActionResult<State> {
    // Collect all paths leading to different destination states with their reward
    // and the probability of activating this path from 'state_id' with 'action_id'.
    let mut dest_states_paths: HashMap<State, Vec<(f64, f64)>> = HashMap::new();

    // Number of cars rented out from first location.
    for l1_out in 0..(l1_day + 1) {
        // The probability of renting all of the cars is the sum of probabilities
        // of number of customers from l1_day to inf.
        let l1_out_prob = if l1_out == l1_day {
            poisson_tail(RENTALS_LAMBDA1, l1_out)
        } else {
            poisson_prob(RENTALS_LAMBDA1, l1_out)
        };

        // Number of cars returned to the first location.
        let max_l1_in = max_cars - l1_day + l1_out;
        for l1_in in 0..(max_l1_in + 1) {
            // The probability of returning cars up to the maximum capacity is the
            // sum of probabilities of number of returns from max_l1_in to inf.
            let l1_in_prob = if l1_in == max_l1_in {
                poisson_tail(RETURNS_LAMBDA1, l1_in)
            } else {
                poisson_prob(RETURNS_LAMBDA1, l1_in)
            };

            // Number of cars rented out from the second location.
            for l2_out in 0..(l2_day + 1) {
                // The probability of renting all of the cars is the sum of probabilities
                // of number of customers from l2_day to inf.
                let l2_out_prob = if l2_out == l2_day {
                    poisson_tail(RENTALS_LAMBDA2, l2_out)
                } else {
                    poisson_prob(RENTALS_LAMBDA2, l2_out)
                };

                // Number of cars returned to the second location.
                let max_l2_in = max_cars - l2_day + l2_out;
                for l2_in in 0..(max_l2_in + 1) {
                    // The probability of returning cars up to the maximum capacity is the
                    // sum of probabilities of number of returns from max_l2_in to inf.
                    let l2_in_prob = if l2_in == max_l2_in {
                        poisson_tail(RETURNS_LAMBDA2, l2_in)
                    } else {
                        poisson_prob(RETURNS_LAMBDA2, l2_in)
                    };

                    let l1_end = l1_day - l1_out + l1_in;
                    let l2_end = l2_day - l2_out + l2_in;
                    assert!(l1_end >= 0 && l1_end <= max_cars);
                    assert!(l2_end >= 0 && l2_end <= max_cars);

                    let probability = l1_out_prob * l1_in_prob * l2_out_prob * l2_in_prob;

                    let reward = ((l1_out + l2_out) as f64) * RENT_REWARD
                        - (transfer.abs() as f64) * TRANSFER_PRICE;

                    let dest_state_id = State::new(l1_end, l2_end);

                    let dest_state_paths = dest_states_paths.entry(dest_state_id).or_default();

                    dest_state_paths.push((reward, probability as f64));
                }
            }
        }
    }

    // Now collect all paths leading to each destination state and compute
    // overall probability of arriving at that state and the mean reward *given*
    // the destination state.
    let mut dest_states = HashMap::new();
    for (dest_state, paths) in dest_states_paths {
        let dest_state_probability = paths.iter().map(|(_r, p)| p).sum();
        assert!(dest_state_probability > 0.0);
        let mean_reward: f64 =
            paths.iter().map(|(r, p)| p * r).sum::<f64>() / dest_state_probability;

        dest_states.insert(
            dest_state,
            ActionDestination {
                probability: dest_state_probability,
                reward: mean_reward,
            },
        );
    }

    // Make sure all probabilities for destination states in the action sum to 1.
    let total_probability: f64 = dest_states
        .iter()
        .map(|(_state_id, dest)| dest.probability)
        .sum();

    assert!(
        (total_probability - 1.0).abs() < 0.1,
        "{}_{}|{}: {}",
        l1_day,
        l2_day,
        transfer,
        total_probability
    );

    ActionResult {
        dest_states: dest_states,
    }
}

pub fn new_car_rental_env(max_cars: i32) -> Env<State, i32> {
    let mut states = HashMap::new();

    let mut actions_cache = HashMap::new();

    // Number of cars on first location at the day end.
    for l1_start in 0..(max_cars + 1) {
        // Number of cars on the second location at the day end.
        for l2_start in 0..(max_cars + 1) {
            let state = State::new(l1_start, l2_start);
            let mut state_actions = StateActions {
                actions: HashMap::new(),
            };

            // Number of cars moved from first to second location
            // (negative for the other way around).
            let min_transfers = -(l2_start.min(MAX_MOVES).min(max_cars - l1_start));
            let max_transfers = l1_start.min(MAX_MOVES).min(max_cars - l2_start);
            for transfer in min_transfers..(max_transfers + 1) {
                let l1_day = l1_start - transfer;
                let l2_day = l2_start + transfer;
                let action_result = actions_cache
                    .entry((l1_day, l2_day))
                    .or_insert_with(|| new_action_result(max_cars, l1_day, l2_day, transfer));

                state_actions
                    .actions
                    .insert(transfer, action_result.clone());
            }

            states.insert(state, state_actions);
        }
    }

    Env { states: states }
}

pub fn new_car_rental_noop_policy(env: &Env<State, i32>) -> Policy<State, i32> {
    let mut policy_states = HashMap::new();
    for (state, _state_actions) in &env.states {
        let mut policy_state_actions = HashMap::new();
        policy_state_actions.insert(0, 1.0);
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

pub fn print_car_rental_policy(policy: &Policy<State, i32>, max_cars: i32) {
    let empty_policy_state = &PolicyState {
        actions: HashMap::new(),
    };
    let mut table = Table::new();
    for r in 0..(max_cars + 1) {
        let mut cells = Vec::new();
        for c in 0..(max_cars + 1) {
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
