use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
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
        for bank in [input_bank, lut_bank, output_bank] {
            if !ctx.cfgs[bank as usize].allocated || ctx.cfgs[bank as usize].cols != 1 {
                panic!("lut: each operand must occupy one allocated bank");
            }
        }
        if iter == 0 || iter > bank_lines() {
            panic!("lut: iter must fit in one bank");
        }

        let pl = pbank(ctx.bank_map, lut_bank);
        let mut table = [0u8; 256];
        table.copy_from_slice(&ctx.banks[pl][..256]);
        let pi = pbank(ctx.bank_map, input_bank);
        let po = pbank(ctx.bank_map, output_bank);
        let (input, output) = ctx.banks.read_write(pi, po);
        for byte in 0..iter * 16 {
            output[byte] = table[input[byte] as usize];
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        if xs2 != 0 || iter == 0 || iter > bank_lines() as u64 {
            panic!("lut: illegal encoding");
        }
        32 + iter * 4
    }
}
