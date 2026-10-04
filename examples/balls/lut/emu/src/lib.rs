pub(crate) use crate::inst::{decode, instruction};

mod lut;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
