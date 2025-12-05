/// Bebop - Accelerator simulator for RISC-V host
///
/// This library provides socket-based communication between host (RISC-V ISA simulator)
/// and custom accelerator implementations.
#[macro_use]
pub mod builtin;
pub mod buckyball;
#[macro_use]
pub mod simulator;

// Re-export log configuration functions for convenience
pub use simulator::utils::log_config::{
    set_forward_log, set_backward_log,
    is_forward_log_enabled, is_backward_log_enabled,
    enable_all_logs, disable_all_logs,
};

pub use buckyball::{NpuConfig, Top};
pub use simulator::{Simulator, StepMode};
