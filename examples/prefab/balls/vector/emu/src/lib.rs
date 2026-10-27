pub(crate) use bebop_bemu::{thread::{instructions::decode, bank_matrix}, ball as instruction};

#[path = "64_vecmat16.rs"]
mod f64_vecmat16;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};

#[bebop_bemu::ball::linkme::distributed_slice(bebop_bemu::BALLS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static BALL: bebop_bemu::Ball = bebop_bemu::Ball { execute: execute_known, cycles: cycles_after_issue };
