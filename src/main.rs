mod blackjack;
mod car_rental;
mod coin_bet;
mod gridworld;
mod solver;

use std::collections::HashMap;

use solver::explicit::*;
use solver::*;

fn main() {
    blackjack::run();
}
