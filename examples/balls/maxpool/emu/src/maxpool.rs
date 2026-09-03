use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::{BallInstruction, ExecContext};

pub struct MaxPool;

impl BallInstruction for MaxPool {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let input_bank = rs1_b0(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        let input_side = (xs2 & 0xf) as usize;
        let output_side = ((xs2 >> 4) & 0xf) as usize;
        let kernel = ((xs2 >> 8) & 0xf) as usize;
        let stride = ((xs2 >> 12) & 0xf) as usize;
        let padding = ((xs2 >> 16) & 0xf) as usize;
        let input_base = ((xs2 >> 20) & 0x3f) as usize;
        let output_base = ((xs2 >> 26) & 0x3f) as usize;
        let output_stride = ((xs2 >> 32) & 0x3f) as usize;
        let start_row = ((xs2 >> 38) & 0xf) as usize;
        let start_col = ((xs2 >> 42) & 0xf) as usize;

        if rs1_b1(xs1) != 0 {
            panic!("maxpool: input bank 1 must be zero");
        }
        if input_bank >= bank_num() as u64 || output_bank >= bank_num() as u64 {
            panic!("maxpool: invalid bank id");
        }
        if input_bank == output_bank {
            panic!("maxpool: input and output banks must differ");
        }
        for bank in [input_bank, output_bank] {
            if !ctx.cfgs[bank as usize].allocated || ctx.cfgs[bank as usize].cols != 1 {
                panic!("maxpool: each operand must occupy one allocated bank");
            }
        }
        if input_side == 0
            || output_side == 0
            || kernel == 0
            || stride == 0
            || output_stride < output_side
            || input_base + input_side * input_side > bank_lines()
            || output_side * output_side != iter
            || output_base + (output_side - 1) * output_stride + output_side > bank_lines()
            || input_side + 2 * padding < kernel + start_row
            || input_side + 2 * padding < kernel + start_col
            || input_side + 2 * padding - kernel - start_row
                != (output_side - 1) * stride
            || input_side + 2 * padding - kernel - start_col
                != (output_side - 1) * stride
        {
            panic!("maxpool: illegal square pooling geometry");
        }

        let pi = pbank(ctx.bank_map, input_bank);
        let po = pbank(ctx.bank_map, output_bank);
        let (input, output) = ctx.banks.read_write(pi, po);
        for output_y in 0..output_side {
            for output_x in 0..output_side {
                let output_offset =
                    (output_base + output_y * output_stride + output_x) * bank_row_bytes();
                let mut maximum = [-128i8; 16];
                for kernel_y in 0..kernel {
                    for kernel_x in 0..kernel {
                        let input_y = (output_y * stride + kernel_y + start_row) as isize
                            - padding as isize;
                        let input_x = (output_x * stride + kernel_x + start_col) as isize
                            - padding as isize;
                        if input_y < 0
                            || input_x < 0
                            || input_y >= input_side as isize
                            || input_x >= input_side as isize
                        {
                            continue;
                        }
                        let input_offset = (input_base
                            + input_y as usize * input_side
                            + input_x as usize)
                            * bank_row_bytes();
                        for lane in 0..16 {
                            maximum[lane] = maximum[lane].max(input[input_offset + lane] as i8);
                        }
                    }
                }
                for lane in 0..16 {
                    output[output_offset + lane] = maximum[lane] as u8;
                }
            }
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        let input_side = xs2 & 0xf;
        let output_side = (xs2 >> 4) & 0xf;
        let kernel = (xs2 >> 8) & 0xf;
        let stride = (xs2 >> 12) & 0xf;
        let padding = (xs2 >> 16) & 0xf;
        let input_base = (xs2 >> 20) & 0x3f;
        let output_base = (xs2 >> 26) & 0x3f;
        let output_stride = (xs2 >> 32) & 0x3f;
        let start_row = (xs2 >> 38) & 0xf;
        let start_col = (xs2 >> 42) & 0xf;
        if rs1_b1(xs1) != 0
            || input_side == 0
            || output_side == 0
            || kernel == 0
            || stride == 0
            || output_stride < output_side
            || input_base + input_side * input_side > bank_lines() as u64
            || output_side * output_side != iter
            || output_base + (output_side - 1) * output_stride + output_side
                > bank_lines() as u64
            || input_side + 2 * padding < kernel + start_row
            || input_side + 2 * padding < kernel + start_col
            || input_side + 2 * padding - kernel - start_row
                != (output_side - 1) * stride
            || input_side + 2 * padding - kernel - start_col
                != (output_side - 1) * stride
        {
            panic!("maxpool: illegal encoding");
        }
        iter * (kernel * kernel * 2 + 2)
    }
}
