pub(crate) use crate::inst::decode;
pub(crate) use crate::inst::instruction;

mod model;
mod quant;

#[cfg(test)]
mod tests;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
