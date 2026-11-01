//! MXINT8QUANT: rows of K FP32 values to packed MXINT8 rows (K INT8 codes,
//! then K/32 E8M0 exponent bytes, zero-padded to whole bank rows).
use bebop_bemu::thread::banks::{bank_lines, bank_row_bytes};
use bebop_bemu::thread::instructions::decode::{
    pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update,
};
use bebop_bemu::ExecContext;

/// Bank rows of one packed row of length k.
fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

/// The block exponent e (stored as e + 127) of 32 FP32 values.
fn block_exponent(values: &[u32]) -> i32 {
    let maximum = values.iter().map(|bits| bits & 0x7fff_ffff).max().unwrap();
    if maximum == 0 {
        return 0;
    }
    let biased = (maximum >> 23) as i32;
    if biased == 0 {
        return -127;
    }
    (biased - 133).max(-127)
}

/// clamp(round_half_even(x / 2^e), -127, 127) of FP32 bits x.
fn encode(bits: u32, exponent: i32) -> u8 {
    let field = ((bits >> 23) & 0xff) as i32;
    let significand = (bits & 0x7f_ffff) | if field == 0 { 0 } else { 0x80_0000 };
    let shift = 150 + exponent - field.max(1);
    assert!(
        shift >= 17,
        "mxint8quantize: element exceeds its block maximum"
    );
    let magnitude = if shift > 31 {
        0
    } else {
        let quotient = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        quotient + u32::from(remainder > half || (remainder == half && quotient & 1 == 1))
    }
    .min(127) as i32;
    (if bits >> 31 == 1 {
        -magnitude
    } else {
        magnitude
    }) as i8 as u8
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let k = (xs2 & 0xffff) as usize;
    let in_base = ((xs2 >> 16) & 0x3fff) as usize;
    let out_base = ((xs2 >> 30) & 0x3fff) as usize;
    assert!(
        rs1_b1(xs1) == 0 && !rs1_read_old(xs1) && !rs1_update(xs1) && xs2 >> 44 == 0,
        "mxint8quantize: reserved fields must be zero"
    );
    assert!(
        rows > 0 && k > 0 && k % 32 == 0,
        "mxint8quantize: rows > 0 and K a positive multiple of 32"
    );
    assert!(
        input != output,
        "mxint8quantize: input and output banks must differ"
    );
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxint8quantize: banks must be allocated with one group"
        );
    }
    assert_eq!(
        bank_row_bytes(),
        16,
        "mxint8quantize: bank rows must be 128 bits"
    );
    let stride = packed_rows(k);
    assert!(
        in_base + rows * k / 4 <= bank_lines() && out_base + rows * stride <= bank_lines(),
        "mxint8quantize: bank window exceeds capacity"
    );
    let source = pbank(ctx, input);
    let values: Vec<u32> = ctx.banks[source][in_base * 16..(in_base + rows * k / 4) * 16]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert!(
        values.iter().all(|bits| bits & 0x7fff_ffff < 0x7f80_0000),
        "mxint8quantize: non-finite input"
    );
    let mut packed = vec![0u8; rows * stride * 16];
    for (row, values) in values.chunks_exact(k).enumerate() {
        let packed = &mut packed[row * stride * 16..(row + 1) * stride * 16];
        for (block, values) in values.chunks_exact(32).enumerate() {
            let exponent = block_exponent(values);
            packed[k + block] = (exponent + 127) as u8;
            for (index, bits) in values.iter().enumerate() {
                packed[block * 32 + index] = encode(*bits, exponent);
            }
        }
    }
    let destination = pbank(ctx, output);
    ctx.banks[destination][out_base * 16..out_base * 16 + packed.len()].copy_from_slice(&packed);
    0
}

// Each block reads eight bank rows, selects an exponent, and writes two code
// rows. One additional scale row is written for every sixteen blocks or tail.
pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let rows = rs1_iter(xs1);
    let blocks = (xs2 & 0xffff) / 32;
    8 + rows * (47 * blocks + 4 * blocks.div_ceil(16))
}
