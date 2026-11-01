pub(crate) use bebop_bemu::ball as instruction;
pub(crate) use bebop_bemu::thread::instructions::decode;

mod model;
mod quant;

mod ball;
mod dispatch;
