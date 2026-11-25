//! The prefab attention thread alone on the framework's thread environment
//! (bebop_bemu::sim::thread_system): the host that loads and launches its
//! program, the TSS and an ideal memory. The Balls its design.pb names are
//! compiled here and register in `bebop_bemu::BALLS`.
include!(concat!(env!("OUT_DIR"), "/balls.rs"));

pub const SIMULATION: bebop_bemu::Simulation = bebop_bemu::Simulation {
    design: include_bytes!(concat!(env!("OUT_DIR"), "/design.pb")),
    instantiate: bebop_bemu::sim::thread_system::instantiate,
};
