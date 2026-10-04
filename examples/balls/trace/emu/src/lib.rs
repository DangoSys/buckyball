pub(crate) use crate::inst::instruction;

#[path = "04_bdb_counter.rs"]
mod f04_bdb_counter;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
