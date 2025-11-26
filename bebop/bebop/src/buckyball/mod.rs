/// Buckyball - Core NPU modules
pub mod frontend;
pub mod memdomain;
pub mod balldomain;
pub mod prototype;
pub mod top;

pub use crate::builtin::{Module, Wire};
pub use frontend::{GlobalDecoder, GlobalReservationStation, DecoderInput, DecoderOutput, MvinConfig, MvoutConfig};
pub use memdomain::{Bank, Controller, DmaOperation, MemDecoder, MemDecoderInput, MemDecoderOutput, MemDomain, MemLoader, MemStorer};
pub use balldomain::BallDomain;
pub use top::Top;
