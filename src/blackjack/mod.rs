use rand::prelude::*;

use prettytable::{Cell, Row, Table};

use crate::solver::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Card {
    Ace,
    Value(u32),
    Face,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Action {
    Hit,
    Stick,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Hand {
    // Value counts usable ace as 11.
    value: u32,
    usable_ace: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct State {
    dealer: Card,
    player: Hand,
}

impl Card {
    pub fn is_ace(&self) -> bool {
        match self {
            Card::Ace => true,
            _ => false,
        }
    }
}

impl Hand {
    fn from_cards(cards: &Vec<Card>) -> Hand {
        let mut hand = Hand::default();
        for c in cards {
            hand = hand.add_card(*c);
        }
        hand
    }

    fn add_card(&self, card: Card) -> Hand {
        let mut hand = *self;
        match card {
            Card::Ace => {
                if !hand.usable_ace && hand.value <= 10 {
                    hand.usable_ace = true;
                    hand.value += 11;
                } else {
                    hand.value += 1;
                }
            }
            Card::Value(v) => hand.value += v,
            Card::Face => hand.value += 10,
        }

        if hand.value > 21 && hand.usable_ace {
            hand.value -= 10;
            hand.usable_ace = false;
        }
        hand
    }

    fn add_random_card(&self) -> Hand {
        self.add_card(random_card())
    }
}

fn random_card() -> Card {
    let r = rand::random::<u32>() % 13 + 1;
    match r {
        1 => Card::Ace,
        2..=10 => Card::Value(r),
        11..=13 => Card::Face,
        _ => panic!(),
    }
}

// Creates an initial random state.
pub fn start_state() -> State {
    State {
        dealer: random_card(),
        player: Hand::default().add_random_card().add_random_card(),
    }
}

// Creates the next state from the current state and action.
// Cards are dealt in random as required per the selected action.
// Returns:
// * (None, 1/-1/0) if the action is Stick; reward is determining by simulating the dealer taking
//   cards until they reach 17.
// * (None, -1) if the action is Hit and the player has gone bust after taking one more card.
// * (Some(State), 0) if the action is Hit and the player didn't go over 21 yet.
pub fn next_state(state: &State, action: &Action) -> (Option<State>, f64) {
    if *action == Action::Stick {
        // Dealer takes cards until they reach 17.
        // Start with 2 cards: hidden (the one that should have been dealt at the beginning,
        // but we only deal it now) and the open card.
        let mut dealer = Hand::default().add_card(state.dealer);
        while dealer.value < 17 {
            dealer = dealer.add_random_card();
        }

        if dealer.value > 21 {
            // Dealer has gone bust.
            return (Option::None, 1.0);
        }

        if state.player.value > dealer.value {
            return (Option::None, 1.0);
        } else if state.player.value < dealer.value {
            return (Option::None, -1.0);
        } else {
            return (Option::None, 0.0);
        }
    }

    // Action is "Hit".
    let player = state.player.add_random_card();
    if player.value > 21 {
        // Player has gone bust.
        return (Option::None, -1.0);
    }

    (
        Option::Some(State {
            dealer: state.dealer,
            player: player,
        }),
        0.0,
    )
}

pub fn random_action(state: &State) -> Action {
    if rand::random::<f64>() < 0.5 {
        Action::Hit
    } else {
        Action::Stick
    }
}

// A policy that only sticks on 20 or higher.
pub fn stick_at_20_policy(state: &State) -> Action {
    if state.player.value < 20 {
        return Action::Hit;
    } else {
        return Action::Stick;
    }
}

pub fn print_policy(policy: &explicit::Policy<State, Action>) {
    let all_cards: Vec<Card> = (2..=10)
        .map(|i| Card::Value(i))
        .chain((&[Card::Ace, Card::Face]).iter().map(|c| *c))
        .collect();

    let mut table = Table::new();

    // Print header.
    let mut header = Vec::new();
    header.push(Cell::new(""));
    header.push(Cell::new("Ace?"));
    for dealer_card in all_cards.iter() {
        header.push(match dealer_card {
            Card::Ace => Cell::new("A"),
            Card::Value(v) => Cell::new(&format!("{}", v)),
            Card::Face => Cell::new("F"),
        });
    }
    table.add_row(Row::new(header));

    for usable_ace in &[false, true] {
        for player_sum in 11..=21 {
            let mut cells = Vec::new();
            cells.push(Cell::new(&format!("{}", player_sum)));
            cells.push(Cell::new(match usable_ace {
                true => "Y",
                false => "N",
            }));
            for dealer_card in all_cards.iter() {
                let state = State {
                    dealer: *dealer_card,
                    player: Hand {
                        usable_ace: *usable_ace,
                        value: player_sum,
                    },
                };

                match policy.states.get(&state) {
                    Some(policy_state) => {
                        let action = policy_state.actions.iter().nth(0).unwrap().0;
                        match action {
                            &Action::Hit => cells.push(Cell::new("H")),
                            &Action::Stick => cells.push(Cell::new("S")),
                        }
                    }
                    None => cells.push(Cell::new("")),
                }
            }
            table.add_row(Row::new(cells));
        }
    }
    table.printstd();
}

pub fn run() {
    // let state_values =
    //     monte_carlo::evaluate_policy(start_state, stick_at_20_policy, next_state, 1.0, 10000000);
    //
    // let mut states_and_values: Vec<(State, f64)> = state_values.into_iter().collect();
    // states_and_values.sort_by(|(k1, v1), (k2, v2)| v2.partial_cmp(v1).unwrap());
    //
    // for (k, v) in states_and_values.iter().take(100) {
    //     println!("{:?}: {}", k, v);
    // }

    let policy = monte_carlo::find_policy(
        &start_state,
        &random_action,
        &next_state,
        1.0,
        0.1,
        10000000,
    );
    print_policy(&policy);
    let policy_functor = monte_carlo::policy_from_explicit(policy);

    // Run simulations.
    let mut total_optimal_returns = 0.0;
    let mut total_naive_returns = 0.0;
    let runs = 100000;
    for _ in 0..runs {
        total_optimal_returns +=
            monte_carlo::run_simulation(&start_state, &policy_functor, &next_state);
        total_naive_returns +=
            monte_carlo::run_simulation(&start_state, &stick_at_20_policy, &next_state);
    }
    println!(
        "Average naive returns: {}",
        (total_naive_returns / runs as f64)
    );
    println!(
        "Average optimal returns: {}",
        (total_optimal_returns / runs as f64)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use Card as C;

    #[test]
    fn hand_value_test() {
        assert_eq!(Hand::from_cards(&vec![C::Ace]).value, 11);
        assert_eq!(Hand::from_cards(&vec![C::Ace, C::Ace]).value, 12);
        assert_eq!(Hand::from_cards(&vec![C::Ace, C::Ace, C::Ace]).value, 13);
        assert_eq!(
            Hand::from_cards(&vec![C::Ace, C::Ace, C::Ace, C::Ace]).value,
            14
        );

        for i in 2..=10 {
            assert_eq!(Hand::from_cards(&vec![C::Value(i)]).value, i);
        }

        assert_eq!(Hand::from_cards(&vec![C::Face]).value, 10);

        assert_eq!(Hand::from_cards(&vec![C::Face, C::Face, C::Ace]).value, 21);
    }
}
