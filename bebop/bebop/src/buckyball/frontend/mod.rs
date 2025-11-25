/// Frontend - Global instruction decoder and reservation station
pub mod global_decoder;
pub mod global_rs;

pub use global_decoder::{Decoder as GlobalDecoder, DecoderInput, DecoderOutput, MvinConfig, MvoutConfig};
pub use global_rs::{GlobalReservationStation, GlobalRsIssue, GlobalRsComplete, RobId};
