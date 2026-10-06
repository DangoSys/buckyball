pub(crate) use crate::inst::{decode, instruction};

mod fp32;
mod mxfp8;
mod smatmul;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
