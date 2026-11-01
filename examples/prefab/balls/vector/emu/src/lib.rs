pub(crate) use bebop_bemu::{
    ball as instruction,
    thread::{bank_matrix, instructions::decode},
};

#[path = "64_vecmat16.rs"]
mod f64_vecmat16;

mod ball;
mod dispatch;
