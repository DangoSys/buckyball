use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn exec_f32_to_i8(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let src = rs1_b0(xs1);
    let unused = rs1_b1(xs1);
    let dst = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if src >= bank_num() as u64 || dst >= bank_num() as u64 {
        panic!("quant_f32_to_i8: invalid bank id");
    }
    if unused != 0 {
        panic!("quant_f32_to_i8: input bank 1 must be zero");
    }
    if src == dst {
        panic!("quant_f32_to_i8: input and output banks must differ");
    }
    if !ctx.cfgs[src as usize].allocated || !ctx.cfgs[dst as usize].allocated {
        panic!("quant_f32_to_i8: bank not allocated");
    }
    if ctx.cfgs[src as usize].cols != 1 || ctx.cfgs[dst as usize].cols != 1 {
        panic!("quant_f32_to_i8: banks must have one column");
    }
    if iter == 0 || iter % 4 != 0 || iter > bank_lines() {
        panic!("quant_f32_to_i8: iter must be a positive multiple of four within bank depth");
    }
    if xs2 >> 32 != 0 {
        panic!("quant_f32_to_i8: rs2[63:32] must be zero");
    }
    let scale = f32::from_bits(xs2 as u32);
    if !scale.is_finite() || scale <= 0.0 {
        panic!("quant_f32_to_i8: scale must be finite and positive");
    }

    let ps = pbank(ctx.bank_map, src);
    let pd = pbank(ctx.bank_map, dst);
    let (input, output) = ctx.banks.read_write(ps, pd);
    for output_row in 0..iter / 4 {
        let mut packed = [0u8; 16];
        for group in 0..4 {
            let input_row = output_row * 4 + group;
            let base = input_row * 16;
            for lane in 0..4 {
                let offset = base + lane * 4;
                let bits = u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                packed[group * 4 + lane] = super::model::quantize_f32(
                    f32::from_bits(bits),
                    scale,
                    false,
                ) as u8;
            }
        }
        let base = output_row * 16;
        output[base..base + 16].copy_from_slice(&packed);
    }
    0
}

pub(crate) fn exec_i32_to_i8(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let src = rs1_b0(xs1);
    let scale_bank = rs1_b1(xs1);
    let dst = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if src >= bank_num() as u64 || scale_bank >= bank_num() as u64 || dst >= bank_num() as u64 {
        panic!("quant_i32_to_i8: invalid bank id");
    }
    if src == scale_bank || src == dst || scale_bank == dst {
        panic!("quant_i32_to_i8: input, scale, and output banks must differ");
    }
    if !ctx.cfgs[src as usize].allocated
        || !ctx.cfgs[scale_bank as usize].allocated
        || !ctx.cfgs[dst as usize].allocated
    {
        panic!("quant_i32_to_i8: bank not allocated");
    }
    if ctx.cfgs[src as usize].cols != 1
        || ctx.cfgs[scale_bank as usize].cols != 1
        || ctx.cfgs[dst as usize].cols != 1
    {
        panic!("quant_i32_to_i8: banks must have one column");
    }
    if iter == 0 || iter % 4 != 0 || iter > bank_lines() {
        panic!("quant_i32_to_i8: iter must be a positive multiple of four within bank depth");
    }
    if xs2 >> 29 != 0 {
        panic!("quant_i32_to_i8: rs2[63:29] must be zero");
    }
    let output_base = ((xs2 >> 1) & 0x7f) as usize;
    let output_width = ((xs2 >> 8) & 0x7f) as usize;
    let output_height = ((xs2 >> 15) & 0x7f) as usize;
    let output_stride = ((xs2 >> 22) & 0x7f) as usize;
    let output_rows = iter / 4;
    if output_width == 0
        || output_height == 0
        || output_stride < output_width
        || output_width.checked_mul(output_height) != Some(output_rows)
        || output_base
            .checked_add((output_height - 1) * output_stride + output_width)
            .is_none_or(|end| end > bank_lines())
    {
        panic!("quant_i32_to_i8: invalid output tile");
    }
    let relu = xs2 & 1 != 0;

    let pscale = pbank(ctx.bank_map, scale_bank);
    let mut scales = [0.0f32; 16];
    for group in 0..4 {
        let base = group * 16;
        for lane in 0..4 {
            let offset = base + lane * 4;
            scales[group * 4 + lane] = f32::from_bits(u32::from_le_bytes(
                ctx.banks[pscale][offset..offset + 4].try_into().unwrap(),
            ));
            if !scales[group * 4 + lane].is_finite() || scales[group * 4 + lane] <= 0.0 {
                panic!("quant_i32_to_i8: scales must be finite and positive");
            }
        }
    }

    let ps = pbank(ctx.bank_map, src);
    let pd = pbank(ctx.bank_map, dst);
    let (input, output) = ctx.banks.read_write(ps, pd);
    for output_row in 0..iter / 4 {
        let mut packed = [0u8; 16];
        for group in 0..4 {
            let input_row = output_row * 4 + group;
            let base = input_row * 16;
            for lane in 0..4 {
                let offset = base + lane * 4;
                let value = i32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                let channel = group * 4 + lane;
                packed[channel] = super::model::quantize_i32(value, scales[channel], relu) as u8;
            }
        }
        let row = output_base
            + (output_row / output_width) * output_stride
            + output_row % output_width;
        let base = row * 16;
        output[base..base + 16].copy_from_slice(&packed);
    }
    0
}

pub(crate) fn f32_to_i8_latency(xs1: u64, xs2: u64) -> u64 {
    let iter = rs1_iter(xs1);
    if iter == 0 || iter % 4 != 0 || xs2 >> 32 != 0 {
        panic!("quant_f32_to_i8: illegal encoding");
    }
    let scale = f32::from_bits(xs2 as u32);
    if !scale.is_finite() || scale <= 0.0 {
        panic!("quant_f32_to_i8: scale must be finite and positive");
    }
    iter
}

pub(crate) fn i32_to_i8_latency(xs1: u64, xs2: u64) -> u64 {
    let iter = rs1_iter(xs1);
    if xs2 >> 29 != 0 {
        panic!("quant_i32_to_i8: illegal encoding");
    }
    let output_base = (xs2 >> 1) & 0x7f;
    let output_width = (xs2 >> 8) & 0x7f;
    let output_height = (xs2 >> 15) & 0x7f;
    let output_stride = (xs2 >> 22) & 0x7f;
    if iter == 0
        || iter % 4 != 0
        || output_width == 0
        || output_height == 0
        || output_stride < output_width
        || output_width * output_height != iter / 4
        || output_base + (output_height - 1) * output_stride + output_width
            > bank_lines() as u64
    {
        panic!("quant_i32_to_i8: illegal encoding");
    }
    iter + 4
}
