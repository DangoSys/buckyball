//! The goban device simulation.
//! Its Hart models register in `bebop_bemu::HARTS`; the Balls its design.pb
//! names are compiled here and register in `bebop_bemu::BALLS`.
extern crate bebop_goban_hart_goban;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

mod simulation;

pub use simulation::SIMULATION;
