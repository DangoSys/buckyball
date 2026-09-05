use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

pub struct Lut;

impl BallInstruction for Lut {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let input_bank = rs1_b0(xs1);
        let lut_bank = rs1_b1(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        if xs2 != 0 {
            panic!("lut: rs2 must be zero");
        }
        if input_bank >= bank_num() as u64
            || lut_bank >= bank_num() as u64
            || output_bank >= bank_num() as u64
        {
            panic!("lut: invalid bank id");
        }
        if input_bank == lut_bank || input_bank == output_bank || lut_bank == output_bank {
            panic!("lut: banks must be distinct");
        }
        if !ctx.cfgs[input_bank as usize].allocated
            || ctx.cfgs[input_bank as usize].cols != 1
            || !ctx.cfgs[output_bank as usize].allocated
            || ctx.cfgs[output_bank as usize].cols != 1
        {
            panic!("lut: input and output must each occupy one allocated bank");
        }
        let lut_cols = ctx.cfgs[lut_bank as usize].cols;
        if !ctx.cfgs[lut_bank as usize].allocated || (lut_cols != 1 && lut_cols != 4) {
            panic!("lut: table must occupy one or four allocated banks");
        }
        if iter == 0 || iter > bank_lines() {
            panic!("lut: iter must fit in one bank");
        }

        let pi = pbank(ctx.bank_map, input_bank);
        let po = pbank(ctx.bank_map, output_bank);
        for row in 0..iter {
            let mut result = [0u8; 16];
            for channel in 0..16 {
                let input = ctx.banks[pi][row * 16 + channel] as usize;
                let flat = if lut_cols == 4 {
                    channel * 256 + input
                } else {
                    input
                };
                let group = flat / 1024;
                let offset = flat % 1024;
                let pl = pbank_group(ctx.bank_map, lut_bank, group as u64);
                result[channel] = ctx.banks[pl][offset];
            }
            ctx.banks[po][row * 16..(row + 1) * 16].copy_from_slice(&result);
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        if xs2 != 0 || iter == 0 || iter > bank_lines() as u64 {
            panic!("lut: illegal encoding");
        }
        2 + iter * 36
    }
}
