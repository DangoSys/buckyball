/// Bebop - Accelerator simulator for RISC-V Spike
///
/// This library provides socket-based communication between Spike (RISC-V ISA simulator)
/// and custom accelerator implementations.
#[macro_use]
pub mod log;
pub mod builtin;
pub mod buckyball;
pub mod config;
pub mod simulator;
pub mod socket;

pub use buckyball::{MemDomain, Module, MvinConfig, MvoutConfig, Top};
pub use config::NpuConfig;
pub use simulator::Simulator;
pub use socket::{DmaClient, SocketMsg, SocketResp, SocketServer};
