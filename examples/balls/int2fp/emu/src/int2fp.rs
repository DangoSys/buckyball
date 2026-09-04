use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input_bank = rs1_b0(xs1);
    let scale_bank = rs1_b1(xs1);
    let output_bank = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if input_bank >= bank_num() as u64
        || scale_bank >= bank_num() as u64
        || output_bank >= bank_num() as u64
    {
        panic!("int32_to_fp32: invalid bank id");
    }
    if input_bank == scale_bank || input_bank == output_bank || scale_bank == output_bank {
        panic!("int32_to_fp32: banks must be distinct");
    }
    if !ctx.cfgs[input_bank as usize].allocated
        || !ctx.cfgs[scale_bank as usize].allocated
        || !ctx.cfgs[output_bank as usize].allocated
    {
        panic!("int32_to_fp32: bank not allocated");
    }
    if ctx.cfgs[input_bank as usize].cols != 1
        || ctx.cfgs[scale_bank as usize].cols != 1
        || ctx.cfgs[output_bank as usize].cols != 1
    {
        panic!("int32_to_fp32: operands must each occupy one bank");
    }
    if iter == 0 || iter % 4 != 0 || iter > bank_lines() {
        panic!("int32_to_fp32: iter must be a positive multiple of four within bank depth");
    }
    if xs2 >> 1 != 0 {
        panic!("int32_to_fp32: rs2[63:1] must be zero");
    }
    let relu = xs2 & 1 != 0;

    let input = pbank(ctx.bank_map, input_bank);
    let scale = pbank(ctx.bank_map, scale_bank);
    let output = pbank(ctx.bank_map, output_bank);
    for row in 0..iter {
        for lane in 0..4 {
            let offset = row * 16 + lane * 4;
            let scale_offset = (row % 4) * 16 + lane * 4;
            let value = i32::from_le_bytes(
                ctx.banks[input][offset..offset + 4].try_into().unwrap(),
            );
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
    0
}

pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let iter = rs1_iter(xs1);
    if iter == 0 || iter % 4 != 0 || xs2 >> 1 != 0 {
        panic!("int32_to_fp32: illegal encoding");
    }
    iter
}
