use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use super::instruction::ExecContext;
use bebop_bemu::thread::banks::bank_lines;

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    assert!(!rs1_read_old(xs1), "int2fp: reads no old destination; rs1[29] must be clear");
    let input_bank = rs1_b0(xs1);
    let scale_bank = rs1_b1(xs1);
    let output_bank = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if input_bank == scale_bank || input_bank == output_bank || scale_bank == output_bank {
        panic!("int32_to_fp32: banks must be distinct");
    }
    if !ctx.config(input_bank).allocated
        || !ctx.config(scale_bank).allocated
        || !ctx.config(output_bank).allocated
    {
        panic!("int32_to_fp32: bank not allocated");
    }
    let groups = ctx.config(input_bank).cols;
    if groups == 0
        || ctx.config(scale_bank).cols != groups
        || ctx.config(output_bank).cols != groups
    {
        panic!("int32_to_fp32: operands must have the same channel groups");
    }
    if iter == 0 || iter % 4 != 0 || iter > bank_lines() {
        panic!("int32_to_fp32: iter must be a positive multiple of four within bank depth");
    }
    if xs2 >> 1 != 0 {
        panic!("int32_to_fp32: rs2[63:1] must be zero");
    }
    let relu = xs2 & 1 != 0;

    // Group g of every operand is channel block g; each block uses the same rows.
    for group in 0..groups {
        let input = pbank_group(ctx, input_bank, group);
        let scale = pbank_group(ctx, scale_bank, group);
        let output = pbank_group(ctx, output_bank, group);
        for row in 0..iter {
            for lane in 0..4 {
                let offset = row * 16 + lane * 4;
                let scale_offset = (row % 4) * 16 + lane * 4;
                let value =
                    i32::from_le_bytes(ctx.banks[input][offset..offset + 4].try_into().unwrap());
                let factor = f32::from_bits(u32::from_le_bytes(
                    ctx.banks[scale][scale_offset..scale_offset + 4]
                        .try_into()
                        .unwrap(),
                ));
                if !factor.is_finite() || factor <= 0.0 {
                    panic!("int32_to_fp32: scales must be finite and positive");
                }
                let value = if relu { value.max(0) } else { value };
                let result = (value as f32) * factor;
                ctx.banks[output][offset..offset + 4].copy_from_slice(&result.to_le_bytes());
            }
        }
    }
    0
}

pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let iter = rs1_iter(xs1);
    if iter == 0 || iter % 4 != 0 || xs2 >> 1 != 0 {
        panic!("int32_to_fp32: illegal encoding");
    }
    iter
}
