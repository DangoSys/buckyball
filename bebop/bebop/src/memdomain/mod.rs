pub mod decoder;
pub mod mem;
pub mod memctrl;
pub mod memdomain;

pub use decoder::{DmaOperation, MemDecoder, MemDecoderInput, MemDecoderOutput};
pub use mem::Bank;
pub use memctrl::Controller;
pub use memdomain::MemDomain;
