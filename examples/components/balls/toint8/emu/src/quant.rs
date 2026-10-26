use bebop_bemu::thread::banks::bank_lines;
use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

pub(crate) fn exec_f32_to_i8(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let src = rs1_b0(xs1);
    let unused = rs1_b1(xs1);
    let dst = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if unused != 0 {
        panic!("quant_f32_to_i8: input bank 1 must be zero");
    }
    if src == dst {
        panic!("quant_f32_to_i8: input and output banks must differ");
    }
    if !ctx.config(src).allocated || !ctx.config(dst).allocated {
        panic!("quant_f32_to_i8: bank not allocated");
    }
    let groups = ctx.config(src).cols;
    if groups == 0 || ctx.config(dst).cols != groups {
        panic!("quant_f32_to_i8: input and output must have the same channel groups");
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

    // Group g of every operand is channel block g; each block uses the same rows.
    for channel_group in 0..groups {
    let ps = pbank_group(ctx, src, channel_group);
    let pd = pbank_group(ctx, dst, channel_group);
    let (input, output) = ctx.banks.read_write(ps, pd);
    for output_row in 0..iter / 4 {
        let mut packed = [0u8; 16];
        for group in 0..4 {
            let input_row = output_row * 4 + group;
            let base = input_row * 16;
            for lane in 0..4 {
                let offset = base + lane * 4;
                let bits = u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                packed[group * 4 + lane] =
                    super::model::quantize_f32(f32::from_bits(bits), scale, false) as u8;
            }
        }
        let base = output_row * 16;
        output[base..base + 16].copy_from_slice(&packed);
    }
    }
    0
}

pub(crate) fn exec_i32_to_i8(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let src = rs1_b0(xs1);
    let scale_bank = rs1_b1(xs1);
    let dst = rs1_b2(xs1);
    let iter = rs1_iter(xs1) as usize;
    if src == scale_bank || src == dst || scale_bank == dst {
        panic!("quant_i32_to_i8: input, scale, and output banks must differ");
    }
    if !ctx.config(src).allocated || !ctx.config(scale_bank).allocated || !ctx.config(dst).allocated
    {
        panic!("quant_i32_to_i8: bank not allocated");
    }
    let groups = ctx.config(src).cols;
    if groups == 0 || ctx.config(scale_bank).cols != groups || ctx.config(dst).cols != groups {
        panic!("quant_i32_to_i8: input, scale, and output must have the same channel groups");
    }
    if iter == 0 || iter % 4 != 0 || iter > bank_lines() {
        panic!("quant_i32_to_i8: iter must be a positive multiple of four within bank depth");
    }
    if xs2 >> 35 != 0 {
        panic!("quant_i32_to_i8: rs2[63:35] must be zero");
    }
    let input_base = ((xs2 >> 29) & 0x3f) as usize;
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
        || input_base + iter > bank_lines()
    {
        panic!("quant_i32_to_i8: invalid output core");
    }
    let relu = xs2 & 1 != 0;

    // Group g of every operand is channel block g; each block uses the same rows.
    for channel_group in 0..groups {
    let pscale = pbank_group(ctx, scale_bank, channel_group);
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

    let ps = pbank_group(ctx, src, channel_group);
    let pd = pbank_group(ctx, dst, channel_group);
    let (input, output) = ctx.banks.read_write(ps, pd);
    for output_row in 0..iter / 4 {
        let mut packed = [0u8; 16];
        for group in 0..4 {
            let input_row = input_base + output_row * 4 + group;
            let base = input_row * 16;
            for lane in 0..4 {
                let offset = base + lane * 4;
                let value = i32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                let channel = group * 4 + lane;
                packed[channel] = super::model::quantize_i32(value, scales[channel], relu) as u8;
            }
        }
        let row =
            output_base + (output_row / output_width) * output_stride + output_row % output_width;
        let base = row * 16;
        output[base..base + 16].copy_from_slice(&packed);
    }
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
    let input_base = (xs2 >> 29) & 0x3f;
    if xs2 >> 35 != 0 {
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
        || output_base + (output_height - 1) * output_stride + output_width > bank_lines() as u64
        || input_base + iter > bank_lines() as u64
    {
        panic!("quant_i32_to_i8: illegal encoding");
    }
    iter + 4
}
