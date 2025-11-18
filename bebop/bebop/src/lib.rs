/// Bebop - Accelerator simulator for RISC-V Spike
///
/// This library provides socket-based communication between Spike (RISC-V ISA simulator)
/// and custom accelerator implementations.
pub mod builtin;
pub mod config;
pub mod global_decoder;
pub mod memdomain;
pub mod simulator;
pub mod socket;
pub mod top;

pub use builtin::{Module, Wire};
pub use config::NpuConfig;
pub use global_decoder::{Decoder, DecoderInput, DecoderOutput, MvinConfig, MvoutConfig};
pub use memdomain::{Bank, Controller, DmaOperation, MemDecoder, MemDecoderInput, MemDecoderOutput, MemDomain};
pub use simulator::Simulator;
pub use socket::{DmaClient, SocketMsg, SocketResp, SocketServer};
pub use top::Top;
