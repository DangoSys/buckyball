pub(crate) use crate::inst::{decode, instruction};

#[path = "55_mxfp2int.rs"]
mod f55_mxfp2int;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
