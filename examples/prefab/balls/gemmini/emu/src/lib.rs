pub(crate) use bebop_bemu::{ball as instruction, ball::blink as bank_matrix, ball::blink as decode};

mod gemmini_config;
mod gemmini_flush;
mod gemmini_preload;
mod gemmini_compute_preloaded;
mod gemmini_compute_accumulated;
mod gemmini_loop_ws;
mod gemmini_loop_conv_ws;
mod gemmini_state;
mod latency;
mod loop_micro_ops;

mod ball;
mod dispatch;
