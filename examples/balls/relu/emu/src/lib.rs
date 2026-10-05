pub(crate) use crate::inst::{decode, instruction};

#[path = "50_relu.rs"]
mod f50_relu;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
