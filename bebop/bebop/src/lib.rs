/// Bebop - Accelerator simulator for RISC-V Spike
///
/// This library provides socket-based communication between Spike (RISC-V ISA simulator)
/// and custom accelerator implementations.
pub mod buckyball;
pub mod config;
pub mod simulator;
pub mod socket;

pub use buckyball::{Bank, Controller, GlobalDecoder, DecoderInput, DecoderOutput, DmaOperation, MemDecoder, MemDecoderInput, MemDecoderOutput, MemDomain, Module, MvinConfig, MvoutConfig, Top, Wire};
pub use config::NpuConfig;
pub use simulator::Simulator;
pub use socket::{DmaClient, SocketMsg, SocketResp, SocketServer};
