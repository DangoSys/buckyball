pub(crate) use crate::inst::{decode, instruction};

mod int2fp;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
