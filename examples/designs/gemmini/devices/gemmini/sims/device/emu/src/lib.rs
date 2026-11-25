//! The Gemmini device simulation.
//! Its Hart model registers in `bebop_bemu::HARTS`.
extern crate bebop_hart_gemmini;

mod simulation;

pub use simulation::SIMULATION;
