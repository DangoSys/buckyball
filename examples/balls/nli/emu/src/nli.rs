use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

/// NliBall - Non-uniform Linear Interpolation. Mirrors the RTL in
/// `arch/src/main/scala/NliBall.scala` bit-for-bit:
///   out = clamp((slope[seg] * x) >> 7 + intercept[seg], -128, 127)
/// where `seg` is the number of cutpoints `<= x` (16 non-uniform segments).
pub struct Nli;

#[inline]
fn mac(slope: i8, x: i8, intercept: i8) -> u8 {
    let prod = (slope as i16) * (x as i16);
    let scaled = prod >> 7; // arithmetic shift, floor toward -inf (== RTL SInt >>)
    let acc = scaled + (intercept as i16);
    let clamped = acc.clamp(-128, 127);
    clamped as u8
}

impl BallInstruction for Nli {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let input_bank = rs1_b0(xs1);
        let table_bank = rs1_b1(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        if xs2 != 0 {
            panic!("nli: rs2 must be zero");
        }
        if input_bank >= bank_num() as u64
            || table_bank >= bank_num() as u64
            || output_bank >= bank_num() as u64
        {
            panic!("nli: invalid bank id");
        }
        if input_bank == table_bank || input_bank == output_bank || table_bank == output_bank {
            panic!("nli: banks must be distinct");
        }
        for bank in [input_bank, table_bank, output_bank] {
            let cfg = &ctx.cfgs[bank as usize];
            if !cfg.allocated || cfg.cols != 1 {
                panic!("nli: input/table/output must each occupy one allocated bank (col=1)");
            }
        }
        if iter == 0 || iter > bank_lines() {
            panic!("nli: iter must fit in one bank");
        }

        let pi = pbank(ctx.bank_map, input_bank);
        let pt = pbank(ctx.bank_map, table_bank);
        let po = pbank(ctx.bank_map, output_bank);

        // Coefficient table: row 0 = 15 cutpoints, row 1 = 16 slopes (Q7),
        // row 2 = 16 intercepts. Copy into owned buffers so the immutable table
        // borrow ends before the mutable output write below.
        let cutpoints: Vec<i8> = ctx.banks[pt][0..15].iter().map(|&b| b as i8).collect();
        let slopes: Vec<i8> = ctx.banks[pt][16..32].iter().map(|&b| b as i8).collect();
        let intercepts: Vec<i8> = ctx.banks[pt][32..48].iter().map(|&b| b as i8).collect();

        for row in 0..iter {
            let mut result = [0u8; 16];
            for lane in 0..16 {
                let x = ctx.banks[pi][row * 16 + lane] as i8;
                let mut seg = 0usize;
                for &c in &cutpoints {
                    if x >= c {
                        seg += 1;
                    }
                }
                result[lane] = mac(slopes[seg], x, intercepts[seg]);
            }
            ctx.banks[po][row * 16..(row + 1) * 16].copy_from_slice(&result);
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        if xs2 != 0 || iter == 0 || iter > bank_lines() as u64 {
            panic!("nli: illegal encoding");
        }
        // 3 coefficient-row reads + per input row (read, compute, write).
        6 + iter * 5
    }
}
