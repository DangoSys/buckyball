//! The poly device simulation.
//! Its Core models register in `bebop_bemu::CORES`; the Balls its design.pb
//! names are compiled here and register in `bebop_bemu::BALLS`.
extern crate bebop_core_reduce;
extern crate bebop_design_poly;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

mod simulation;

pub use simulation::SIMULATION;
