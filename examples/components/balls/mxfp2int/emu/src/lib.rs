pub(crate) use bebop_bemu::{thread::instructions::decode, ball as instruction};

#[path = "55_mxfp2int.rs"]
mod f55_mxfp2int;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};

#[bebop_bemu::ball::linkme::distributed_slice(bebop_bemu::BALLS)]
#[linkme(crate = bebop_bemu::ball::linkme)]
static BALL: bebop_bemu::Ball = bebop_bemu::Ball { execute: execute_known, cycles: cycles_after_issue };
