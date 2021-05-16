use std::collections::HashMap;

use plotlib::{
    page::Page,
    repr::Plot,
    style::{LineJoin, LineStyle, PointMarker, PointStyle},
    view::ContinuousView,
};
use prettytable::{Cell, Row, Table};

use crate::solver::*;

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
