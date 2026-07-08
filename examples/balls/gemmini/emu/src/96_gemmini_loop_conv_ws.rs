//===- 96_gemmini_loop_conv_ws.rs - GEMMINI_LOOP_CONV_WS instruction -------===//

use super::gemmini_state::{gemini, mem_i8, mem_write_i32};
use super::instruction::{ExecContext, Instruction};

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

fn exec_loop_impl(memory: &mut [u8]) -> u64 {
    let st = gemini().lock().unwrap().loop_conv.clone();
    let in_ch = st.in_ch as usize;
    let out_ch = st.out_ch as usize;
    if in_ch == 0 || out_ch == 0 {
        panic!("gemmini_loop_conv_ws: zero channels");
    }
    let in0 = st.addr_input;
    let w0 = st.addr_weight;
    let out0 = st.addr_output;

    for j in 0..out_ch {
        let mut acc = 0i32;
        for k in 0..in_ch {
            let a = mem_i8(memory, in0 + k as u64);
            let w = mem_i8(memory, w0 + (k * out_ch + j) as u64);
            acc += a as i32 * w as i32;
        }
        mem_write_i32(memory, out0 + j as u64 * 4, acc);
    }
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
    fn exec(_xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        exec_loop_impl(ctx.memory)
    }
    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        latency_impl(Self::FUNCT)
    }
}
