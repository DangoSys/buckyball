pub(crate) use crate::inst::{decode, instruction};

mod accumulator;
mod fp32;
mod mxfp8;
mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
