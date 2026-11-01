//! The piplus Core models, registered in CORES by component.
mod compute;
mod control;
mod shared_mem;

pub use compute::ComputeCore;
pub use control::ControlCore;
