//! The boom hart simulation: the boom hart alone in the framework's
//! single-hart system. Its Hart model registers in `bebop_bemu::HARTS`.
extern crate bebop_hart_boom;

include!(concat!(env!("OUT_DIR"), "/balls.rs"));

pub const SIMULATION: bebop_bemu::Simulation = bebop_bemu::Simulation {
    design: include_bytes!(concat!(env!("OUT_DIR"), "/design.pb")),
    instantiate: bebop_bemu::sim::hart_system::instantiate,
};
