//! The poly device: its own Core model; the cores are instantiated by component on the CBus.
mod communication;
pub mod core;

pub use communication::instantiate;
