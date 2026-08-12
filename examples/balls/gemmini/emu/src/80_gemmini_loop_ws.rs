//===- 80_gemmini_loop_ws.rs - GEMMINI_LOOP_WS instruction -----------------===//

use super::super::bank::{bank_row_bytes, MATRIX_SIZE};
use super::gemmini_state::gemini;
use super::instruction::{ExecContext, Instruction};
use super::loop_micro_ops::{
    alloc, checked_stride, compute, free_after_digest, mvin, mvout, preload,
};

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

fn checked_tile_addr(
    base: u64,
    outer: u64,
    outer_stride: u64,
    inner: u64,
    inner_stride: u64,
) -> u64 {
    let outer_offset = outer
        .checked_mul(outer_stride)
        .expect("gemmini_loop_ws: outer address offset overflow");
    let inner_offset = inner
        .checked_mul(inner_stride)
        .expect("gemmini_loop_ws: inner address offset overflow");
    base.checked_add(outer_offset)
        .and_then(|addr| addr.checked_add(inner_offset))
        .expect("gemmini_loop_ws: address overflow")
}

fn exec_loop_impl(xs2: u64, ctx: &mut ExecContext) -> u64 {
    let g = gemini().lock().unwrap();
    let lw = g.loop_ws.clone();
    let dataflow = g.cfg.dataflow;
    drop(g);

    let bank_a = xs2 & 0x3ff;
    let bank_b = (xs2 >> 10) & 0x3ff;
    let bank_c = (xs2 >> 20) & 0x3ff;
    let low_d = ((xs2 >> 30) & 1) != 0;
    assert!(
        bank_a != bank_b && bank_a != bank_c && bank_b != bank_c,
        "gemmini_loop_ws: banks must be distinct"
    );
    assert!(
        lw.max_i > 0 && lw.max_j > 0 && lw.max_k > 0,
        "gemmini_loop_ws: bounds must be > 0"
    );
    assert_eq!(
        dataflow, 0,
        "gemmini_loop_ws: current RTL CISC contract supports OS dataflow only"
    );
    assert!(
        low_d && lw.addr_d == 0 && lw.stride_d == 0,
        "gemmini_loop_ws: only zero D is supported"
    );

    let dim = MATRIX_SIZE as u64;
    let bank_bytes = bank_row_bytes() as u64;
    let stride_a = checked_stride(lw.stride_a, bank_bytes, "gemmini_loop_ws stride_a");
    let stride_b = checked_stride(lw.stride_b, bank_bytes, "gemmini_loop_ws stride_b");
    let stride_c = checked_stride(
        lw.stride_c,
        bank_bytes
            .checked_mul(4)
            .expect("gemmini_loop_ws: output row width overflow"),
        "gemmini_loop_ws stride_c",
    );

    alloc(ctx, bank_a, 1);
    alloc(ctx, bank_b, 1);
    alloc(ctx, bank_c, 4);

    mvin(ctx, bank_a, lw.addr_a, dim, stride_a);
    mvin(ctx, bank_b, lw.addr_b, dim, stride_b);

    for i in 0..lw.max_i {
        for j in 0..lw.max_j {
            for k in 0..lw.max_k {
                preload(ctx, bank_a, bank_c, dim);
                compute(ctx, k != 0, bank_a, bank_b, bank_c, dim, false, false);

                let addr_c = checked_tile_addr(lw.addr_c, i, lw.stride_c, j, dim * 4);
                mvout(ctx, bank_c, addr_c, dim, stride_c);

                let next = if k + 1 < lw.max_k {
                    Some((i, j, k + 1))
                } else if j + 1 < lw.max_j {
                    Some((i, j + 1, 0))
                } else if i + 1 < lw.max_i {
                    Some((i + 1, 0, 0))
                } else {
                    None
                };
                if let Some((next_i, next_j, next_k)) = next {
                    let addr_a = checked_tile_addr(lw.addr_a, next_i, lw.stride_a, next_k, dim);
                    let addr_b = checked_tile_addr(lw.addr_b, next_k, lw.stride_b, next_j, dim);
                    mvin(ctx, bank_a, addr_a, dim, stride_a);
                    mvin(ctx, bank_b, addr_b, dim, stride_b);
                }
            }
        }
    }

    free_after_digest(ctx, bank_a);
    free_after_digest(ctx, bank_b);
    free_after_digest(ctx, bank_c);
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
    fn exec(_xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        exec_loop_impl(xs2, ctx)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}
