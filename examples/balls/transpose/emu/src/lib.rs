pub(crate) use crate::inst::{decode, instruction};

mod transpose;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
