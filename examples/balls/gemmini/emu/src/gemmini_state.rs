//! Global config for Gemmini / loop instructions. Mutex is fine because Spike calls in on a single worker.
use super::super::bank::{mem_read, mem_write};
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

pub fn mem_u8(mem: &[u8], addr: u64) -> u8 {
    mem_read(mem, addr)
}

pub fn mem_i8(mem: &[u8], addr: u64) -> i8 {
    mem_u8(mem, addr) as i8
}

pub fn mem_i32_le(mem: &[u8], addr: u64) -> i32 {
    let mut b = [0u8; 4];
    for (i, byte) in b.iter_mut().enumerate() {
        *byte = mem_read(mem, addr + i as u64);
    }
    i32::from_le_bytes(b)
}

pub fn mem_write_i32(mem: &mut [u8], addr: u64, v: i32) {
    let b = v.to_le_bytes();
    for (i, byte) in b.iter().enumerate() {
        mem_write(mem, addr + i as u64, *byte);
    }
}
