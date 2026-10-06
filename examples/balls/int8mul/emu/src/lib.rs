pub(crate) use crate::inst::{decode, instruction};

mod int8mul;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
