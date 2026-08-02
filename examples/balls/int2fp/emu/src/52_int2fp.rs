//===- 52_int2fp.rs - INT2FP instruction (INT to FP32 / INT8) --------------===//

use super::super::bank::{bank_num, bank_size};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

mod model;

pub struct Int2Fp;

impl Instruction for Int2Fp {
    const FUNCT: u32 = 52;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let src = rs1_b0(xs1);
        let dst = rs1_b2(xs1);
        let depth = rs1_iter(xs1) as usize;

        if src >= bank_num() as u64 || dst >= bank_num() as u64 {
            panic!("int2fp: invalid bank_id");
        }

        if depth == 0 {
            panic!("int2fp: iter must be > 0");
        }

        let sc = ctx.cfgs[src as usize];
        let dc = ctx.cfgs[dst as usize];
        if !sc.allocated || !dc.allocated {
            panic!("int2fp: bank not allocated");
        }

        let scale_bits = (xs2 & 0xffff_ffff) as u32;
        let output_mode = ((xs2 >> 32) & 0x3) as u32;
        if output_mode > 1 {
            panic!("int2fp: reserved output mode {output_mode}");
        }

        match (output_mode, sc.cols, dc.cols) {
            (0, 1, 1) => {
                let ps = pbank(ctx.bank_map, src);
                let pd = pbank(ctx.bank_map, dst);
                for i in 0..depth {
                    let base = i * 16;
                    if base + 16 > bank_size() {
                        panic!("int2fp: out of range");
                    }
                    for lane in 0..4 {
                        let off = base + lane * 4;
                        let v = i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                        let o = model::int2fp_fp32_bits(v, scale_bits);
                        ctx.banks[pd][off..off + 4].copy_from_slice(&o.to_le_bytes());
                    }
                }
            }
            (0, 1, 4) => {
                let ps = pbank(ctx.bank_map, src);
                for i in 0..depth {
                    let src_base = i * 16;
                    let dst_base = i * 16;
                    if src_base + 16 > bank_size() || dst_base + 16 > bank_size() {
                        panic!("int2fp: out of range");
                    }
                    for j in 0..16 {
                        let v = ctx.banks[ps][src_base + j] as i8;
                        let o = model::int2fp_fp32_bits(i32::from(v), scale_bits);
                        let group = j / 4;
                        let lane = j % 4;
                        let pd = pbank_group(ctx.bank_map, dst, group as u64);
                        let off = dst_base + lane * 4;
                        ctx.banks[pd][off..off + 4].copy_from_slice(&o.to_le_bytes());
                    }
                }
            }
            (0, 2, 2) | (0, 4, 4) => {
                let groups = sc.cols as u64;
                for group in 0..groups {
                    let ps = pbank_group(ctx.bank_map, src, group);
                    let pd = pbank_group(ctx.bank_map, dst, group);
                    for i in 0..depth {
                        let base = i * 16;
                        if base + 16 > bank_size() {
                            panic!("int2fp: out of range");
                        }
                        for lane in 0..4 {
                            let off = base + lane * 4;
                            let v =
                                i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                            let o = model::int2fp_fp32_bits(v, scale_bits);
                            ctx.banks[pd][off..off + 4].copy_from_slice(&o.to_le_bytes());
                        }
                    }
                }
            }
            (1, 4, 1) => {
                let pd = pbank(ctx.bank_map, dst);
                for i in 0..depth {
                    let base = i * 16;
                    if base + 16 > bank_size() {
                        panic!("int2fp: out of range");
                    }
                    for group in 0..4 {
                        let ps = pbank_group(ctx.bank_map, src, group);
                        for lane in 0..4 {
                            let off = base + lane * 4;
                            let v =
                                i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                            let q = model::int2fp_i8_bits(v, scale_bits);
                            ctx.banks[pd][base + group as usize * 4 + lane] = q as u8;
                        }
                    }
                }
            }
            _ => {
                panic!(
                    "int2fp: unsupported mode/layout output_mode={} src_cols={} dst_cols={}",
                    output_mode, sc.cols, dc.cols
                );
            }
        }
        0
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1).max(1)
    }
}
