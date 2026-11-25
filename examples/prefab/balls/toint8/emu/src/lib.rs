pub(crate) use bebop_bemu::ball as instruction;
pub(crate) use bebop_bemu::ball::blink as decode;

mod model;
mod quant;

mod ball;
mod dispatch;
mod latency;
