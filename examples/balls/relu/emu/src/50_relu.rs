//===- 50_relu.rs - RELU instruction ----------------------------------------===//

use super::super::bank::{bank_num, bank_size};
use super::decode::{pbank, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

pub struct Relu;

impl BallInstruction for Relu {
    fn exec(xs1: u64, _xs2: u64, ctx: &mut ExecContext) -> u64 {
        let src = rs1_b0(xs1);
        let dst = rs1_b2(xs1);
        let depth = rs1_iter(xs1) as usize;

        if src >= bank_num() as u64 || dst >= bank_num() as u64 {
            panic!("relu: invalid bank_id");
        }

        let sc = ctx.cfgs[src as usize];
        let dc = ctx.cfgs[dst as usize];
        if !sc.allocated || !dc.allocated {
            panic!("relu: bank not allocated");
        }

        let ps = pbank(ctx.bank_map, src);
        let pd = pbank(ctx.bank_map, dst);

        if sc.cols == 1 && dc.cols == 1 {
            for i in 0..depth {
                let base = i * 16;
                if base + 16 > bank_size() {
                    panic!("relu: out of range");
                }
                for j in 0..16 {
                    let v = ctx.banks[ps][base + j] as i8;
                    ctx.banks[pd][base + j] = if v < 0 { 0 } else { v as u8 };
                }
            }
            return 0;
        }

        if sc.cols == 4 && dc.cols == 4 {
            for i in 0..depth {
                let base = i * 64;
                if base + 64 > bank_size() {
                    panic!("relu: out of range");
                }
                for j in 0..16 {
                    let off = base + j * 4;
                    let v = i32::from_le_bytes(ctx.banks[ps][off..off + 4].try_into().unwrap());
                    let o = if v < 0 { 0 } else { v };
                    ctx.banks[pd][off..off + 4].copy_from_slice(&o.to_le_bytes());
                }
            }
            return 0;
        }

        panic!(
            "relu: unsupported layout src_cols={} dst_cols={}",
            sc.cols, dc.cols
        );
    }

    fn latency(xs1: u64, _xs2: u64) -> u64 {
        rs1_iter(xs1).max(1)
    }
}
