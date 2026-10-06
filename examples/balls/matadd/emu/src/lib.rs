pub(crate) use crate::inst::{decode, instruction};

#[path = "64_matadd.rs"]
mod f64_matadd;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
