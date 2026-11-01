pub(crate) use bebop_bemu::{ball as instruction, thread::instructions::decode};

#[path = "50_relu.rs"]
mod f50_relu;

mod ball;
mod dispatch;
