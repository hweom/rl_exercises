use std::collections::HashMap;

use plotlib::{
    page::Page,
    repr::Plot,
    style::{LineJoin, LineStyle, PointMarker, PointStyle},
    view::ContinuousView,
};
use prettytable::{Cell, Row, Table};

use crate::solver::{explicit::*, *};

const LIMIT: i32 = 100;

pub fn new_coin_env(heads_prob: f64) -> Env<i32, i32> {
    let mut states = HashMap::new();

    // Loop over current amount of money.
    for money in 1..LIMIT {
        let mut actions = HashMap::new();
        // Loop over possible bets.
        let max_bet = money.min(LIMIT - money);
        for bet in 0..(max_bet + 1) {
            let mut action_dests = HashMap::new();
            // Win destination.
            action_dests.insert(
                (money + bet).min(LIMIT),
                ActionDestination {
                    probability: heads_prob,
                    reward: if money + bet >= LIMIT { 1.0 } else { 0.0 },
                },
            );
            // Lose destination.
            action_dests.insert(
                (money - bet).max(0),
                ActionDestination {
                    probability: 1.0 - heads_prob,
                    reward: 0.0,
                },
            );

            actions.insert(
                bet,
                ActionResult {
                    dest_states: action_dests,
                },
            );
        }

        states.insert(money, StateActions { actions: actions });
    }

    // Add final states.
    states.insert(0, StateActions::default());
    states.insert(LIMIT, StateActions::default());

    Env { states: states }
}

pub fn make_cautious_policy() -> Policy<i32, i32> {
    let policy_states = (1..LIMIT)
        .map(|i| {
            let mut actions = HashMap::new();
            actions.insert(1, 1.0);
            (i, PolicyState { actions: actions })
        })
        .collect();

    Policy {
        states: policy_states,
    }
}

pub fn print_coin_state_values(state_values: &HashMap<i32, f64>) {
    let values = (1..LIMIT)
        .map(|i| (i as f64, *state_values.get(&i).unwrap_or(&0.0)))
        .collect();
    let s1 = Plot::new(values).point_style(PointStyle::new().marker(PointMarker::Circle));
    let v = ContinuousView::new()
        .add(s1)
        .x_range(0.0, 100.0)
        .x_label("State")
        .y_label("Value");
    println!(
        "{}",
        Page::single(&v).dimensions(100, 50).to_text().unwrap()
    );
}

pub fn print_coin_policy(policy: &Policy<i32, i32>) {
    let values: Vec<(f64, f64)> = (1..LIMIT)
        .map(|i| {
            let policy_actions = &policy.states.get(&i).unwrap().actions;

            let min_bet = policy_actions.keys().max().unwrap();

            (i as f64, (*min_bet) as f64)
        })
        .collect();

    // Output to screen.
    let s1 = Plot::new(values.clone()).point_style(PointStyle::new().marker(PointMarker::Circle));
    let v = ContinuousView::new()
        .add(s1)
        .x_range(0.0, 100.0)
        .x_label("State")
        .y_label("Action");
    println!(
        "{}",
        Page::single(&v).dimensions(100, 50).to_text().unwrap()
    );
}

pub fn run() {
    // Create environment.
    println!("Creating environment");
    let env = new_coin_env(0.49);

    let mut state_values = HashMap::new();
    for i in 0..100000 {
        let (new_state_values, delta) = iterate_state_value(&env, &state_values, 1.0);
        state_values = new_state_values;
        println!("Delta: {}", delta);
        if delta < 0.0000001 {
            break;
        }
    }

    print_coin_state_values(&state_values);

    let uniform_policy = make_uniform_policy(&env);
    let cautious_policy = make_cautious_policy();
    let optimal_policy = make_greedy_policy(&env, &state_values, 1.0);
    print_coin_policy(&optimal_policy);

    let simulations = 100000;
    let mut uniform_reward = 0.0;
    let mut cautious_reward = 0.0;
    let mut optimal_reward = 0.0;
    let start_state = 10;
    for _ in 0..simulations {
        uniform_reward = uniform_reward + run_simulation(&env, &uniform_policy, start_state, 1000);
        cautious_reward =
            cautious_reward + run_simulation(&env, &cautious_policy, start_state, 1000);
        optimal_reward = optimal_reward + run_simulation(&env, &optimal_policy, start_state, 1000);
    }
    println!(
        "Average uniform reward: {}",
        uniform_reward / simulations as f64
    );
    println!(
        "Average cautious reward: {}",
        cautious_reward / simulations as f64
    );
    println!(
        "Average optimal reward: {}",
        optimal_reward / simulations as f64
    );
}
