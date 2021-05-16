mod car_rental;
mod coin_bet;
mod gridworld;
mod solver;

use std::collections::HashMap;

use car_rental::*;
use coin_bet::*;
use gridworld::*;
use solver::*;

fn car_rental() {
    // Create environment.
    println!("Creating environment");
    let env = new_car_rental_env(20);

    // Create policy.
    println!("Creating intial policy");
    let mut policy = new_car_rental_noop_policy(&env);
    let mut state_values = HashMap::new();

    for i in 0..5 {
        println!("Evaluating policy");
        for i in 0..10000 {
            let (new_state_values, delta) = evaluate_policy(&env, &policy, &state_values, 0.9);
            state_values = new_state_values;
            if i % 10 == 0 {
                println!("{}: delta {}", i, delta);
            }
            if delta < 0.0001 {
                break;
            }
        }
        println!("done!");

        policy = make_greedy_policy(&env, &state_values, 0.9);
        print_car_rental_policy(&policy, 20);
    }
}

fn coin_bet() {
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

fn main() {
    coin_bet();
}
