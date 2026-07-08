//===- 80_gemmini_loop_ws.rs - GEMMINI_LOOP_WS instruction -----------------===//

use super::gemmini_state::{gemini, in_shift as apply_in_shift, mem_i32_le, mem_i8, mem_write_i32};
use super::instruction::{ExecContext, Instruction};

// Shared implementation
fn exec_cfg_impl(funct: u32, xs2: u64) -> u64 {
    let mut g = gemini().lock().unwrap();
    match funct {
        80 => {
            g.loop_ws.max_k = xs2 & 0xffff;
            g.loop_ws.max_j = (xs2 >> 16) & 0xffff;
            g.loop_ws.max_i = (xs2 >> 32) & 0xffff;
        }
        81 => g.loop_ws.addr_a = xs2 & ((1u64 << 39) - 1),
        82 => g.loop_ws.addr_b = xs2 & ((1u64 << 39) - 1),
        83 => g.loop_ws.addr_d = xs2 & ((1u64 << 39) - 1),
        84 => g.loop_ws.addr_c = xs2 & ((1u64 << 39) - 1),
        85 => {
            g.loop_ws.stride_a = xs2 & 0xffff_ffff;
            g.loop_ws.stride_b = xs2 >> 32;
        }
        86 => {
            g.loop_ws.stride_d = xs2 & 0xffff_ffff;
            g.loop_ws.stride_c = xs2 >> 32;
        }
        _ => panic!("gemmini_loop_ws: unknown cfg funct={funct}"),
    }
    0
}

fn exec_loop_impl(memory: &mut [u8]) -> u64 {
    let g = gemini().lock().unwrap();
    let lw = g.loop_ws.clone();
    let a_transpose = g.cfg.a_transpose;
    let b_transpose = g.cfg.b_transpose;
    let shift = g.cfg.in_shift;
    drop(g);

    let n = lw.stride_a as usize;
    if n == 0 || n > 64 {
        panic!("gemmini_loop_ws: bad stride/n");
    }

    for i in 0..n {
        for j in 0..n {
            let ii = i as u64;
            let jj = j as u64;
            let mut acc = if lw.addr_d == 0 {
                0i32
            } else {
                let off = lw.addr_d + ii * lw.stride_d + jj * 4;
                mem_i32_le(memory, off)
            };
            for k in 0..n {
                let kk = k as u64;
                let av = if a_transpose {
                    mem_i8(memory, lw.addr_a + kk * lw.stride_a + ii)
                } else {
                    mem_i8(memory, lw.addr_a + ii * lw.stride_a + kk)
                };
                let bv = if b_transpose {
                    mem_i8(memory, lw.addr_b + jj * lw.stride_b + kk)
                } else {
                    mem_i8(memory, lw.addr_b + kk * lw.stride_b + jj)
                };
                acc += av as i32 * bv as i32;
            }
            let out = apply_in_shift(acc, shift);
            let c_off = lw.addr_c + ii * lw.stride_c + jj * 4;
            mem_write_i32(memory, c_off, out);
        }
    }
    0
}

fn latency_impl(funct: u32) -> u64 {
    if funct == 87 {
        256
    } else {
        1
    }
}

// Individual instruction types for each funct
pub struct GemminiLoopWsConfigBounds;
impl Instruction for GemminiLoopWsConfigBounds {
    const FUNCT: u32 = 80;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigAddrA;
impl Instruction for GemminiLoopWsConfigAddrA {
    const FUNCT: u32 = 81;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigAddrB;
impl Instruction for GemminiLoopWsConfigAddrB {
    const FUNCT: u32 = 82;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigAddrD;
impl Instruction for GemminiLoopWsConfigAddrD {
    const FUNCT: u32 = 83;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigAddrC;
impl Instruction for GemminiLoopWsConfigAddrC {
    const FUNCT: u32 = 84;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigStridesAB;
impl Instruction for GemminiLoopWsConfigStridesAB {
    const FUNCT: u32 = 85;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWsConfigStridesDC;
impl Instruction for GemminiLoopWsConfigStridesDC {
    const FUNCT: u32 = 86;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopWs;
impl Instruction for GemminiLoopWs {
    const FUNCT: u32 = 87;
    fn exec(_xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        exec_loop_impl(ctx.memory)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}
