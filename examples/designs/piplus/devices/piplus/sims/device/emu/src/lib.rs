//! The piplus device simulation.
//! Its Hart models register in `bebop_bemu::HARTS`; the Balls its design.pb
//! names are compiled here and register in `bebop_bemu::BALLS`.
extern crate bebop_piplus_hart_control;
extern crate bebop_piplus_hart_dense;
extern crate bebop_piplus_hart_action;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

pub const SIMULATION: bebop_bemu::Simulation = bebop_bemu::Simulation {
    design: include_bytes!(concat!(env!("OUT_DIR"), "/design.pb")),
    instantiate: bebop_bemu::sim::device_system::instantiate,
};
