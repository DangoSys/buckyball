//===- lib.rs - Toy BEMU instruction set ----------------------------------===//

pub use super::bank_matrix;
pub use super::decode;
pub use super::instruction;

#[path = "../../../../balls/trace/emu/src/04_bdb_counter.rs"]
pub mod f04_bdb_counter;
#[path = "../../../../balls/im2col/emu/src/48_im2col.rs"]
pub mod f48_im2col;
#[path = "../../../../balls/transpose/emu/src/49_transpose.rs"]
pub mod f49_transpose;
#[path = "../../../../balls/relu/emu/src/50_relu.rs"]
pub mod f50_relu;
#[path = "../../../../balls/fp2int/emu/src/51_fp2int.rs"]
pub mod f51_fp2int;
#[path = "../../../../balls/int2fp/emu/src/52_int2fp.rs"]
pub mod f52_int2fp;
#[path = "../../../../balls/gemmini/emu/src/02_gemmini_config.rs"]
pub mod f02_gemmini_config;
#[path = "../../../../balls/gemmini/emu/src/03_gemmini_flush.rs"]
pub mod f03_gemmini_flush;
#[path = "../../../../balls/gemmini/emu/src/53_gemmini_preload.rs"]
pub mod f53_gemmini_preload;
#[path = "../../../../balls/mxfp2int/emu/src/55_mxfp2int.rs"]
pub mod f55_mxfp2int;
#[path = "../../../../balls/vector/emu/src/64_mul_warp16.rs"]
pub mod f64_mul_warp16;
#[path = "../../../../balls/systolicarray/emu/src/65_bfp.rs"]
pub mod f65_bfp;
#[path = "../../../../balls/gemmini/emu/src/66_gemmini_compute_preloaded.rs"]
pub mod f66_gemmini_compute_preloaded;
#[path = "../../../../balls/gemmini/emu/src/67_gemmini_compute_accumulated.rs"]
pub mod f67_gemmini_compute_accumulated;
#[path = "../../../../balls/gemmini/emu/src/80_gemmini_loop_ws.rs"]
pub mod f80_gemmini_loop_ws;
#[path = "../../../../balls/gemmini/emu/src/96_gemmini_loop_conv_ws.rs"]
pub mod f96_gemmini_loop_conv_ws;
#[path = "../../../../balls/gemmini/emu/src/gemmini_state.rs"]
pub mod gemmini_state;

use instruction::{ExecContext, Instruction};

macro_rules! register_instructions {
    ($($inst:path),* $(,)?) => {
        pub fn execute_known(
            funct: u32,
            xs1: u64,
            xs2: u64,
            ctx: &mut ExecContext,
        ) -> Option<u64> {
            match funct {
                $(
                    <$inst as Instruction>::FUNCT => {
                        Some(<$inst as Instruction>::exec(xs1, xs2, ctx))
                    }
                )*
                _ => None,
            }
        }

        pub fn cycles_after_issue(funct: u32, xs1: u64, xs2: u64) -> u64 {
            match funct {
                $(
                    <$inst as Instruction>::FUNCT => {
                        <$inst as Instruction>::latency(xs1, xs2)
                    }
                )*
                _ => 1,
            }
        }
    };
}

register_instructions! {
    super::f00_fence::Fence,
    super::f01_barrier::Barrier,
    f02_gemmini_config::GemminiConfig,
    f03_gemmini_flush::GemminiFlush,
    f04_bdb_counter::BdbCounter,
    super::f16_mvout::Mvout,
    super::f32_mset::Mset,
    super::f33_mvin::Mvin,
    super::f34_mmio_set::MmioSet,
    super::f35_mvin_mmio::MvinMmio,
    f48_im2col::Im2col,
    f49_transpose::Transpose,
    f50_relu::Relu,
    f51_fp2int::Fp2Int,
    f52_int2fp::Int2Fp,
    f53_gemmini_preload::GemminiPreload,
    f55_mxfp2int::Mxfp2Int,
    f64_mul_warp16::MulWarp16,
    f65_bfp::Bfp,
    f66_gemmini_compute_preloaded::GemminiComputePreloaded,
    f67_gemmini_compute_accumulated::GemminiComputeAccumulated,
    f80_gemmini_loop_ws::GemminiLoopWsConfigBounds,
    f80_gemmini_loop_ws::GemminiLoopWsConfigAddrA,
    f80_gemmini_loop_ws::GemminiLoopWsConfigAddrB,
    f80_gemmini_loop_ws::GemminiLoopWsConfigAddrD,
    f80_gemmini_loop_ws::GemminiLoopWsConfigAddrC,
    f80_gemmini_loop_ws::GemminiLoopWsConfigStridesAB,
    f80_gemmini_loop_ws::GemminiLoopWsConfigStridesDC,
    f80_gemmini_loop_ws::GemminiLoopWs,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig1,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig2,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig3,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig4,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig5,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig6,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig7,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig8,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWsConfig9,
    f96_gemmini_loop_conv_ws::GemminiLoopConvWs,
}
