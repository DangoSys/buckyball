//===- 51_quant.rs - FP2INT instruction (FP32 to INT quantization) ----------------------===//

use super::super::bank::{BANK_NUM, BANK_SIZE};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

mod model;

pub struct Fp2Int;

impl Instruction for Fp2Int {
    const FUNCT: u32 = 51;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let src = rs1_b0(xs1);
        let dst = rs1_b2(xs1);
        let depth = rs1_iter(xs1) as usize;

        if src >= BANK_NUM as u64 || dst >= BANK_NUM as u64 {
            panic!("fp2int: invalid bank_id");
        }

        if depth == 0 {
            panic!("fp2int: iter must be > 0");
        }

        let sc = ctx.cfgs[src as usize];
        let dc = ctx.cfgs[dst as usize];
        if !sc.allocated || !dc.allocated {
            panic!("fp2int: bank not allocated");
        }

        let ps = pbank(ctx.bank_map, src);
        let pd = pbank(ctx.bank_map, dst);
        let scale_bits = (xs2 & 0xffff_ffff) as u32;

        // Support two modes:
        // 1. FP32 -> INT32: src_cols=1, dst_cols=1 (4 bytes -> 4 bytes)
        // 2. FP32 -> INT8: src_cols=4, dst_cols=1 (4 bytes -> 1 byte, with clamping)
        match (sc.cols, dc.cols) {
            (1, 1) => {
                // FP32 -> INT32 mode
                for i in 0..depth {
                    let src_base = i * 64;
                    let dst_base = i * 64;
                    if src_base + 64 > BANK_SIZE || dst_base + 64 > BANK_SIZE {
                        panic!("fp2int: out of range");
                    }
                    for j in 0..16 {
                        let off = src_base + j * 4;
                        let fp_bits = u32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                        let q = model::fp2int_i32_bits(fp_bits, scale_bits);
                        let dst_off = dst_base + j * 4;
                        ctx.banks[pd][dst_off..dst_off + 4].copy_from_slice(&q.to_le_bytes());
                    }
                }
            }
            (4, 1) => {
                // FP32 -> INT8 mode
                for i in 0..depth {
                    let src_base = i * 16;
                    let dst_base = i * 16;
                    if src_base + 16 > BANK_SIZE || dst_base + 16 > BANK_SIZE {
                        panic!("fp2int: out of range");
                    }
                    for group in 0..4 {
                        let ps = pbank_group(ctx.bank_map, src, group);
                        for lane in 0..4 {
                            let off = src_base + lane * 4;
                            let fp_bits = u32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                            let q = model::fp2int_i8_bits(fp_bits, scale_bits);
                            ctx.banks[pd][dst_base + group as usize * 4 + lane] = q as u8;
                        }
                    }
                }
            }
            _ => {
                panic!(
                    "fp2int: unsupported layout src_cols={} dst_cols={}. Supported: (1,1) for FP32->INT32, (4,1) for FP32->INT8",
                    sc.cols, dc.cols
                );
            }
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1).max(1)
    }
}
