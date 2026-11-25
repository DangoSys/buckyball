//! The goban hart simulation: the goban hart and its accelerator alone in the
//! framework's single-hart system. Its Hart model registers in
//! `bebop_bemu::HARTS`; the Balls its design.pb names are compiled here and
//! register in `bebop_bemu::BALLS`.
extern crate bebop_goban_hart_goban;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

pub const SIMULATION: bebop_bemu::Simulation = bebop_bemu::Simulation {
    design: include_bytes!(concat!(env!("OUT_DIR"), "/design.pb")),
    instantiate: bebop_bemu::sim::hart_system::instantiate,
};
