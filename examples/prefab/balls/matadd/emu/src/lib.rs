pub(crate) use bebop_bemu::{ball as instruction, thread::instructions::decode};

#[path = "64_matadd.rs"]
mod f64_matadd;

mod ball;
mod dispatch;
