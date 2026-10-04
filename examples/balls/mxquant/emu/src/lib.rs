pub(crate) use crate::inst::{decode, instruction};

mod dispatch;
mod quant;

pub use dispatch::{cycles_after_issue, execute_known};
