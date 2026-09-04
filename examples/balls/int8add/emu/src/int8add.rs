use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub struct Int8Add;

impl Int8Add {
    pub fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext, relu: bool) -> u64 {
        let lhs_bank = rs1_b0(xs1);
        let rhs_bank = rs1_b1(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        let lhs_ratio = f32::from_bits(xs2 as u32);
        let rhs_ratio = f32::from_bits((xs2 >> 32) as u32);

        if lhs_bank >= bank_num() as u64 || rhs_bank >= bank_num() as u64 || output_bank >= bank_num() as u64 {
            panic!("int8add: invalid bank id");
        }
        if lhs_bank == rhs_bank || lhs_bank == output_bank || rhs_bank == output_bank {
            panic!("int8add: banks must be distinct");
        }
        if iter == 0 || iter > bank_lines() {
            panic!("int8add: iter must fit in one physical bank");
        }
        if !lhs_ratio.is_finite() || lhs_ratio <= 0.0 || !rhs_ratio.is_finite() || rhs_ratio <= 0.0 {
            panic!("int8add: ratios must be finite and positive");
        }
        let lhs = &ctx.cfgs[lhs_bank as usize];
        let rhs = &ctx.cfgs[rhs_bank as usize];
        let output = &ctx.cfgs[output_bank as usize];
        if !lhs.allocated || !rhs.allocated || !output.allocated {
            panic!("int8add: all banks must be allocated");
        }
        if lhs.cols == 0 || lhs.cols != rhs.cols || lhs.cols != output.cols {
            panic!("int8add: bank groups must match");
        }

        for group in 0..lhs.cols as usize {
            let pl = pbank_group(ctx.bank_map, lhs_bank, group as u64);
            let pr = pbank_group(ctx.bank_map, rhs_bank, group as u64);
            let po = pbank_group(ctx.bank_map, output_bank, group as u64);
            for row in 0..iter {
                let base = row * bank_row_bytes();
                let mut lhs_row = [0u8; 16];
                let mut rhs_row = [0u8; 16];
                lhs_row.copy_from_slice(&ctx.banks[pl][base..base + 16]);
                rhs_row.copy_from_slice(&ctx.banks[pr][base..base + 16]);
                for lane in 0..16 {
                    let value = (lhs_row[lane] as i8 as f32) * lhs_ratio
                        + (rhs_row[lane] as i8 as f32) * rhs_ratio;
                    let rounded = value.round_ties_even();
                    let clamped = if relu { rounded.max(0.0) } else { rounded }.clamp(-128.0, 127.0);
                    ctx.banks[po][base + lane] = clamped as i8 as u8;
                }
            }
        }
        0
    }

    pub fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        let lhs_ratio = f32::from_bits(xs2 as u32);
        let rhs_ratio = f32::from_bits((xs2 >> 32) as u32);
        if iter == 0 || iter > bank_lines() as u64 {
            panic!("int8add: iter must fit in one physical bank");
        }
        if !lhs_ratio.is_finite() || lhs_ratio <= 0.0 || !rhs_ratio.is_finite() || rhs_ratio <= 0.0 {
            panic!("int8add: ratios must be finite and positive");
        }
        iter * 4
    }
}
