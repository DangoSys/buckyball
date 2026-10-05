pub(crate) use crate::inst::{bank_matrix, decode, instruction};

#[path = "02_gemmini_config.rs"]
mod f02_gemmini_config;
#[path = "03_gemmini_flush.rs"]
mod f03_gemmini_flush;
#[path = "53_gemmini_preload.rs"]
mod f53_gemmini_preload;
#[path = "66_gemmini_compute_preloaded.rs"]
mod f66_gemmini_compute_preloaded;
#[path = "67_gemmini_compute_accumulated.rs"]
mod f67_gemmini_compute_accumulated;
#[path = "80_gemmini_loop_ws.rs"]
mod f80_gemmini_loop_ws;
#[path = "96_gemmini_loop_conv_ws.rs"]
mod f96_gemmini_loop_conv_ws;
mod gemmini_state;
mod loop_micro_ops;

mod dispatch;

pub use dispatch::{cycles_after_issue, execute_known};
