pub(crate) use crate::inst::{bank_matrix, decode, instruction};

#[path = "64_vecmat16.rs"]
mod f64_vecmat16;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
