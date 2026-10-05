pub(crate) use crate::inst::{decode, instruction};

mod maxpool;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
