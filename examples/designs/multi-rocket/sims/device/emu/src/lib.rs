//! The multi-rocket device simulation.
//! Its Core models register in `bebop_bemu::CORES`; the Balls its design.pb
//! names are compiled here and register in `bebop_bemu::BALLS`.
extern crate bebop_core_rocket;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

pub const SIMULATION: bebop_bemu::Simulation = bebop_bemu::Simulation {
    design: include_bytes!("../../../../output/generated/design.pb"),
    instantiate: bebop_bemu::device::cbus::instantiate,
};
