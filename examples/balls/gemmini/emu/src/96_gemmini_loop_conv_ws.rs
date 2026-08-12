//===- 96_gemmini_loop_conv_ws.rs - GEMMINI_LOOP_CONV_WS instruction -------===//

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
        96 => {
            g.loop_conv.batch = xs2 & 0xffff;
            g.loop_conv.in_dim = (xs2 >> 16) & 0xffff;
            g.loop_conv.in_ch = (xs2 >> 32) & 0xffff;
        }
        97 => {
            g.loop_conv.out_ch = xs2 & 0xffff;
            g.loop_conv.out_dim = (xs2 >> 16) & 0xffff;
            g.loop_conv.stride = (xs2 >> 32) & 0xff;
            g.loop_conv.padding = (xs2 >> 40) & 0xff;
        }
        98 => {
            g.loop_conv.kernel_dim = xs2 & 0xff;
            g.loop_conv.pool_size = (xs2 >> 8) & 0xff;
            g.loop_conv.pool_stride = (xs2 >> 16) & 0xff;
            g.loop_conv.pool_padding = (xs2 >> 24) & 0xff;
        }
        99 => g.loop_conv.addr_bias = xs2 & ((1u64 << 39) - 1),
        100 => g.loop_conv.addr_input = xs2 & ((1u64 << 39) - 1),
        101 => g.loop_conv.addr_weight = xs2 & ((1u64 << 39) - 1),
        102 => g.loop_conv.addr_output = xs2 & ((1u64 << 39) - 1),
        103 => {
            g.loop_conv.input_stride = xs2 & 0xffff_ffff;
            g.loop_conv.weight_stride = xs2 >> 32;
        }
        104 => g.loop_conv.output_stride = xs2 & 0xffff_ffff,
        _ => panic!("gemmini_loop_conv_ws: unknown cfg funct={funct}"),
    }
    0
}

fn checked_mul(lhs: u64, rhs: u64, name: &str) -> u64 {
    lhs.checked_mul(rhs)
        .unwrap_or_else(|| panic!("gemmini_loop_conv_ws: {name} overflow"))
}

fn checked_add(lhs: u64, rhs: u64, name: &str) -> u64 {
    lhs.checked_add(rhs)
        .unwrap_or_else(|| panic!("gemmini_loop_conv_ws: {name} overflow"))
}

fn input_addr(
    st: &super::gemmini_state::LoopConvCfg,
    batch: u64,
    irow: u64,
    icol: u64,
    kch: u64,
) -> u64 {
    let spatial = checked_add(
        checked_mul(
            batch,
            checked_mul(st.in_dim, st.in_dim, "input plane"),
            "input batch",
        ),
        checked_add(
            checked_mul(irow, st.in_dim, "input row"),
            icol,
            "input column",
        ),
        "input spatial offset",
    );
    let element = checked_add(
        checked_mul(spatial, st.in_ch, "input channels"),
        kch,
        "input channel tile",
    );
    checked_add(st.addr_input, element, "input address")
}

fn weight_addr(
    st: &super::gemmini_state::LoopConvCfg,
    krow: u64,
    kcol: u64,
    kch: u64,
    och: u64,
) -> u64 {
    let kernel = checked_add(
        checked_mul(krow, st.kernel_dim, "weight kernel row"),
        kcol,
        "weight kernel column",
    );
    let input_channel = checked_add(
        checked_mul(kernel, st.in_ch, "weight input channels"),
        kch,
        "weight kch",
    );
    let element = checked_add(
        checked_mul(input_channel, st.out_ch, "weight output channels"),
        och,
        "weight och",
    );
    checked_add(st.addr_weight, element, "weight address")
}

fn output_addr(
    st: &super::gemmini_state::LoopConvCfg,
    batch: u64,
    orow: u64,
    ocol: u64,
    och: u64,
) -> u64 {
    let spatial = checked_add(
        checked_mul(
            batch,
            checked_mul(st.out_dim, st.out_dim, "output plane"),
            "output batch",
        ),
        checked_add(
            checked_mul(orow, st.out_dim, "output row"),
            ocol,
            "output column",
        ),
        "output spatial offset",
    );
    let element = checked_add(
        checked_mul(spatial, st.out_ch, "output channels"),
        och,
        "output och",
    );
    checked_add(
        st.addr_output,
        checked_mul(element, 4, "output accumulator bytes"),
        "output address",
    )
}

fn exec_loop_impl(xs2: u64, ctx: &mut ExecContext) -> u64 {
    let st = gemini().lock().unwrap().loop_conv.clone();
    let dataflow = gemini().lock().unwrap().cfg.dataflow;
    let bank_input = xs2 & 0x3ff;
    let bank_weight = (xs2 >> 10) & 0x3ff;
    let bank_output = (xs2 >> 20) & 0x3ff;
    let no_bias = ((xs2 >> 30) & 1) != 0;

    assert!(
        bank_input != bank_weight && bank_input != bank_output && bank_weight != bank_output,
        "gemmini_loop_conv_ws: banks must be distinct"
    );
    assert_eq!(
        dataflow, 1,
        "gemmini_loop_conv_ws: current RTL CISC contract requires WS dataflow"
    );
    assert!(
        st.batch > 0 && st.in_dim > 0 && st.out_dim > 0,
        "gemmini_loop_conv_ws: dimensions must be > 0"
    );
    assert!(
        st.in_ch > 0 && st.out_ch > 0 && st.kernel_dim > 0,
        "gemmini_loop_conv_ws: channels/kernel must be > 0"
    );
    assert!(
        st.stride > 0,
        "gemmini_loop_conv_ws: convolution stride must be > 0"
    );
    assert!(
        no_bias && st.addr_bias == 0,
        "gemmini_loop_conv_ws: bias is not implemented by the RTL CISC path"
    );
    assert!(
        st.pool_size == 0 && st.pool_stride == 0 && st.pool_padding == 0,
        "gemmini_loop_conv_ws: pooling is not implemented by the RTL CISC path"
    );

    let dim = MATRIX_SIZE as u64;
    assert_eq!(
        st.in_ch % dim,
        0,
        "gemmini_loop_conv_ws: input channels must be a multiple of {dim}"
    );
    assert_eq!(
        st.out_ch % dim,
        0,
        "gemmini_loop_conv_ws: output channels must be a multiple of {dim}"
    );
    let bank_bytes = bank_row_bytes() as u64;
    let input_stride = checked_stride(
        st.input_stride,
        bank_bytes,
        "gemmini_loop_conv_ws input_stride",
    );
    let weight_stride = checked_stride(
        st.out_ch,
        bank_bytes,
        "gemmini_loop_conv_ws weight tile stride",
    );
    let output_stride = checked_stride(
        st.output_stride,
        checked_mul(bank_bytes, 4, "output row bytes"),
        "gemmini_loop_conv_ws output_stride",
    );

    alloc(ctx, bank_input, 1);
    alloc(ctx, bank_weight, 1);
    alloc(ctx, bank_output, 4);

    for batch in 0..st.batch {
        for orow in 0..st.out_dim {
            for ocol in 0..st.out_dim {
                for och in (0..st.out_ch).step_by(dim as usize) {
                    for krow in 0..st.kernel_dim {
                        for kcol in 0..st.kernel_dim {
                            for kch in (0..st.in_ch).step_by(dim as usize) {
                                let unpadded_row =
                                    checked_add(checked_mul(orow, st.stride, "irow"), krow, "irow");
                                let unpadded_col =
                                    checked_add(checked_mul(ocol, st.stride, "icol"), kcol, "icol");
                                let is_padding = unpadded_row < st.padding
                                    || unpadded_col < st.padding
                                    || unpadded_row - st.padding >= st.in_dim
                                    || unpadded_col - st.padding >= st.in_dim;
                                if !is_padding {
                                    let irow = unpadded_row - st.padding;
                                    let icol = unpadded_col - st.padding;
                                    mvin(
                                        ctx,
                                        bank_input,
                                        input_addr(&st, batch, irow, icol, kch),
                                        1,
                                        input_stride,
                                    );
                                }
                                mvin(
                                    ctx,
                                    bank_weight,
                                    weight_addr(&st, krow, kcol, kch, och),
                                    dim,
                                    weight_stride,
                                );
                                preload(ctx, bank_weight, bank_output, dim);
                                let accumulated = krow != 0 || kcol != 0 || kch != 0;
                                compute(
                                    ctx,
                                    accumulated,
                                    bank_input,
                                    bank_weight,
                                    bank_output,
                                    dim,
                                    no_bias,
                                    true,
                                );
                            }
                        }
                    }
                    mvout(
                        ctx,
                        bank_output,
                        output_addr(&st, batch, orow, ocol, och),
                        1,
                        output_stride,
                    );
                }
            }
        }
    }

    free_after_digest(ctx, bank_input);
    free_after_digest(ctx, bank_weight);
    free_after_digest(ctx, bank_output);
    0
}

fn latency_impl(funct: u32) -> u64 {
    if funct == 105 {
        256
    } else {
        1
    }
}

// Individual instruction types for each funct
pub struct GemminiLoopConvWsConfig1;
impl Instruction for GemminiLoopConvWsConfig1 {
    const FUNCT: u32 = 96;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig2;
impl Instruction for GemminiLoopConvWsConfig2 {
    const FUNCT: u32 = 97;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig3;
impl Instruction for GemminiLoopConvWsConfig3 {
    const FUNCT: u32 = 98;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig4;
impl Instruction for GemminiLoopConvWsConfig4 {
    const FUNCT: u32 = 99;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig5;
impl Instruction for GemminiLoopConvWsConfig5 {
    const FUNCT: u32 = 100;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig6;
impl Instruction for GemminiLoopConvWsConfig6 {
    const FUNCT: u32 = 101;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig7;
impl Instruction for GemminiLoopConvWsConfig7 {
    const FUNCT: u32 = 102;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig8;
impl Instruction for GemminiLoopConvWsConfig8 {
    const FUNCT: u32 = 103;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWsConfig9;
impl Instruction for GemminiLoopConvWsConfig9 {
    const FUNCT: u32 = 104;
    fn exec(_xs1: u64, xs2: u64, _ctx: &mut ExecContext) -> u64 {
        exec_cfg_impl(Self::FUNCT, xs2)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}

pub struct GemminiLoopConvWs;
impl Instruction for GemminiLoopConvWs {
    const FUNCT: u32 = 105;
    fn exec(_xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        exec_loop_impl(xs2, ctx)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}
