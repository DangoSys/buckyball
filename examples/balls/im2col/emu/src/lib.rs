pub(crate) use crate::inst::{decode, instruction};

mod im2col;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
