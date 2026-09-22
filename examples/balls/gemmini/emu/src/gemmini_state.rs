//! Global config for Gemmini / loop instructions. Mutex is fine because Spike calls in on a single worker.
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Default)]
pub struct GemminiCfg {
    pub dataflow: u8,
    pub a_transpose: bool,
    pub b_transpose: bool,
    pub in_shift: u32,
}

#[derive(Clone, Default)]
pub struct LoopWsCfg {
    pub max_i: u64,
    pub max_j: u64,
    pub max_k: u64,
    pub addr_a: u64,
    pub addr_b: u64,
    pub addr_d: u64,
    pub addr_c: u64,
    pub stride_a: u64,
    pub stride_b: u64,
    pub stride_d: u64,
    pub stride_c: u64,
}

#[derive(Clone, Default)]
pub struct LoopConvCfg {
    pub batch: u64,
    pub in_dim: u64,
    pub in_ch: u64,
    pub out_ch: u64,
    pub out_dim: u64,
    pub stride: u64,
    pub padding: u64,
    pub kernel_dim: u64,
    pub pool_size: u64,
    pub pool_stride: u64,
    pub pool_padding: u64,
    pub addr_bias: u64,
    pub addr_input: u64,
    pub addr_weight: u64,
    pub addr_output: u64,
    pub input_stride: u64,
    pub weight_stride: u64,
    pub output_stride: u64,
}

#[derive(Default)]
pub struct GemminiState {
    pub cfg: GemminiCfg,
    pub loop_ws: LoopWsCfg,
    pub loop_conv: LoopConvCfg,
    /// WS preload: B weights (iter × 16) i8
    pub ws_b: Option<Vec<Vec<i8>>>,
}

static GEMINI: OnceLock<Mutex<GemminiState>> = OnceLock::new();

pub fn gemini() -> &'static Mutex<GemminiState> {
    GEMINI.get_or_init(|| Mutex::new(GemminiState::default()))
}

pub fn in_shift(v: i32, shift: u32) -> i32 {
    if shift == 0 {
        return v;
    }
    if v >= 0 {
        let x = v as u32;
        let point_five = (x >> (shift - 1)) & 1;
        let zeros = if shift <= 1 {
            0
        } else if (x & ((1u32 << (shift - 1)) - 1)) != 0 {
            1
        } else {
            0
        };
        let ones_digit = (x >> shift) & 1;
        let r = point_five & (zeros | ones_digit);
        return ((x >> shift) + r) as i32;
    }
    let x = v as u32;
    let point_five = (x >> (shift - 1)) & 1;
    let zeros = if shift <= 1 {
        0
    } else if (x & ((1u32 << (shift - 1)) - 1)) != 0 {
        1
    } else {
        0
    };
    let ones_digit = (x >> shift) & 1;
    let r = (point_five & (zeros | ones_digit)) != 0;
    let base = v >> shift;
    if r {
        base + 1
    } else {
        base
    }
}
