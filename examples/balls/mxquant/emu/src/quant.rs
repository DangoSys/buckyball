use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use crate::bank::{bank_lines, bank_row_bytes};

fn rounded_shift(value: u32, shift: u32) -> u32 {
    if shift > 31 {
        return 0;
    }
    let half = 1 << (shift - 1);
    (value + half - 1 + ((value >> shift) & 1)) >> shift
}

fn encode(bits: u32, block_exponent: i32) -> u8 {
    let sign = ((bits >> 24) & 128) as u8;
    if bits & 0x7fffffff == 0 {
        return sign;
    }
    let mut exponent = ((bits >> 23) & 255) as i32;
    let mut fraction = bits & 0x7fffff;
    if exponent == 0 {
        let shift = fraction.leading_zeros() - 8;
        fraction = (fraction << shift) & 0x7fffff;
        exponent = 1 - shift as i32;
    }
    exponent -= block_exponent + 120;
    let code = if exponent <= 0 {
        rounded_shift(fraction | 0x800000, (21 - exponent) as u32)
    } else {
        exponent as u32 * 8 + rounded_shift(fraction, 20)
    };
    sign | code.min(126) as u8
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let count = rs1_iter(xs1) as usize;
    assert!(
        xs2 == 0 && rs1_b1(xs1) == 0,
        "mxquant: reserved fields must be zero"
    );
    assert!(
        count > 0 && count % 32 == 0,
        "mxquant: count must be a positive multiple of 32"
    );
    assert!(
        input != output,
        "mxquant: input and output banks must differ"
    );
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxquant: banks must be allocated with one column"
        );
    }
    let bytes = bank_lines() * bank_row_bytes();
    assert!(
        bank_row_bytes() == 16 && count <= bytes / 4 && count + count / 32 <= bytes,
        "mxquant: bank footprint exceeds capacity"
    );
    let source = &ctx.banks[pbank(ctx, input)][..count * 4];
    assert!(
        source
            .chunks_exact(4)
            .all(
                |element| u32::from_le_bytes(element.try_into().unwrap()) & 0x7fffffff < 0x7f800000
            ),
        "mxquant: non-finite activation"
    );
    let mut packed = vec![0u8; count + count / 32];
    for (block, values) in source.chunks_exact(128).enumerate() {
        let maximum = values
            .chunks_exact(4)
            .map(|element| u32::from_le_bytes(element.try_into().unwrap()) & 0x7fffffff)
            .max()
            .unwrap();
        let exponent = if maximum == 0 {
            0
        } else {
            ((maximum >> 23) as i32 - 135).max(-127)
        };
        packed[count + block] = (exponent + 127) as u8;
        for (index, element) in values.chunks_exact(4).enumerate() {
            packed[block * 32 + index] =
                encode(u32::from_le_bytes(element.try_into().unwrap()), exponent);
        }
    }
    let destination = pbank(ctx, output);
    ctx.banks[destination][..packed.len()].copy_from_slice(&packed);
    0
}
