use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use super::instruction::{BallModel, ExecContext};
use bebop_bemu::ball::blink::bank_lines;

pub struct Lut;

impl BallModel for Lut {
    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        assert!(
            !rs1_read_old(xs1),
            "lut: reads no old destination; rs1[29] must be clear"
        );
        let input_bank = rs1_b0(xs1);
        let lut_bank = rs1_b1(xs1);
        let output_bank = rs1_b2(xs1);
        let iter = rs1_iter(xs1) as usize;
        let mode = xs2 & 3;
        let multiplier = xs2 >> 32;
        assert!(mode <= 2 && xs2 & 0xffff_fffc == 0, "lut: reserved encoding");
        assert!(
            if mode == 2 {
                multiplier > 0 && multiplier < 1 << 30
            } else {
                multiplier == 0
            },
            "lut: illegal multiplier"
        );
        if input_bank == lut_bank || input_bank == output_bank || lut_bank == output_bank {
            panic!("lut: banks must be distinct");
        }
        if !ctx.config(input_bank).allocated
            || ctx.config(input_bank).cols != 1
            || !ctx.config(output_bank).allocated
            || ctx.config(output_bank).cols != 1
        {
            panic!("lut: input and output must each occupy one allocated bank");
        }
        let lut = *ctx.config(lut_bank);
        let lut_cols = lut.cols;
        if !lut.allocated || (lut_cols != 1 && lut_cols != 4) {
            panic!("lut: table must occupy one or four allocated banks");
        }
        if iter == 0 || iter > bank_lines() {
            panic!("lut: iter must fit in one bank");
        }

        let pi = pbank(ctx, input_bank);
        let po = pbank(ctx, output_bank);
        if mode != 0 {
            assert!(
                lut_cols == 1 && bank_lines() >= 256,
                "lut: I32 table requires 256 rows in one group"
            );
            let pt = pbank(ctx, lut_bank);
            let input: Vec<i32> = ctx.banks[pi][..iter * 16]
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            let mut output = vec![0u8; iter * 16];
            if mode == 1 {
                for (value, word) in input.iter().zip(output.chunks_exact_mut(4)) {
                    let index = ((*value).clamp(-512, 511) as u32 & 1023) as usize;
                    word.copy_from_slice(&ctx.banks[pt][index * 4..index * 4 + 4]);
                }
            } else {
                let maximum = i64::from(*input.iter().max().unwrap());
                let values: Vec<u64> = input
                    .iter()
                    .map(|value| {
                        let index = (maximum - i64::from(*value)).min(1023) as usize;
                        let value = u32::from_le_bytes(ctx.banks[pt][index * 4..index * 4 + 4].try_into().unwrap());
                        assert!(
                            value <= 1 << 30 && (index != 0 || value == 1 << 30),
                            "lut: invalid normalized table"
                        );
                        u64::from(value)
                    })
                    .collect();
                let sum: u64 = values.iter().sum();
                let reciprocal = (multiplier << 32) / sum;
                for (value, byte) in values.into_iter().zip(&mut output) {
                    let product = value * reciprocal;
                    let quotient = product >> 48;
                    let remainder = product & ((1 << 48) - 1);
                    let rounded =
                        quotient + u64::from(remainder > 1 << 47 || (remainder == 1 << 47 && quotient & 1 != 0));
                    *byte = rounded.min(127) as u8;
                }
            }
            ctx.banks[po][..iter * 16].copy_from_slice(&output);
            ctx.config_mut(output_bank).valid_rows = iter as u64;
            return 0;
        }
        for row in 0..iter {
            let mut result = [0u8; 16];
            for channel in 0..16 {
                let input = ctx.banks[pi][row * 16 + channel] as usize;
                let flat = if lut_cols == 4 { channel * 256 + input } else { input };
                let group = flat / 1024;
                let offset = flat % 1024;
                let pl = pbank_group(ctx, lut_bank, group as u64);
                result[channel] = ctx.banks[pl][offset];
            }
            ctx.banks[po][row * 16..(row + 1) * 16].copy_from_slice(&result);
        }
        0
    }

    fn latency(xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(xs1);
        let mode = xs2 & 3;
        if mode > 2 || xs2 & 0xffff_fffc != 0 || iter == 0 || iter > bank_lines() as u64 {
            panic!("lut: illegal encoding");
        }
        super::latency::cycles(iter, xs2)
    }
}
