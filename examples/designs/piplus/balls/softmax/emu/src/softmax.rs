//! SOFTMAX: the softmax of FP32 rows scaled by a power of two, operation for
//! operation in FP32 as
//! workloads/isa/softmax.h specifies it; the reference RTL is compared with.
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::ExecContext;

/// Bank rows of one row of length w.
fn row_lines(w: usize) -> usize {
    w.div_ceil(4)
}

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let w = (xs2 & 0xffff) as usize;
    let in_base = ((xs2 >> 16) & 0x3fff) as usize;
    let out_base = ((xs2 >> 30) & 0x3fff) as usize;
    let scale_log2 = ((xs2 >> 44) & 0xff) as u8 as i8 as i32;
    assert!(
        rs1_b1(xs1) == 0 && !rs1_read_old(xs1) && !rs1_update(xs1) && xs2 >> 52 == 0,
        "softmax: reserved fields must be zero"
    );
    assert!(scale_log2 >= -126, "softmax: scale exponent below -126");
    let scale = f32::from_bits(((scale_log2 + 127) as u32) << 23);
    assert!(
        rows > 0 && (16..=2048).contains(&w),
        "softmax: rows > 0 and 16 <= W <= 2048"
    );
    assert!(input != output, "softmax: input and output banks must differ");
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "softmax: banks must be allocated with one group"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "softmax: bank rows must be 128 bits");
    let lines = row_lines(w);
    assert!(
        in_base + rows * lines <= bank_lines() && out_base + rows * lines <= bank_lines(),
        "softmax: bank window exceeds capacity"
    );
    let source = pbank(ctx, input);
    let mut result = vec![0u8; rows * lines * 16];
    for row in 0..rows {
        let at = (in_base + row * lines) * 16;
        let words = ctx.banks[source][at..at + w * 4].as_chunks::<4>().0;
        let scaled = |i: usize| f32::from_le_bytes(words[i]) * scale;
        let mut finite = false;
        for i in 0..w {
            let v = scaled(i);
            assert!(
                v.is_finite() || v == f32::NEG_INFINITY,
                "softmax: a value neither finite nor -inf"
            );
            finite |= v.is_finite();
        }
        assert!(finite, "softmax: a row without a finite value");
        let offset = row * lines * 16;
        let out = result[offset..offset + 4 * w].as_chunks_mut::<4>().0;
        let sum = super::latency::exponentials(w, scaled, |i, e| out[i] = e.to_le_bytes());
        let inverse = 1.0 / sum;
        for word in out {
            *word = (f32::from_le_bytes(*word) * inverse).to_le_bytes();
        }
    }
    let destination = pbank(ctx, output);
    ctx.banks[destination][out_base * 16..out_base * 16 + result.len()].copy_from_slice(&result);
    0
}

/// The cycles of the RTL (latency.rs), on the input rows it reads.
pub(crate) fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let words = ctx.banks[pbank(ctx, rs1_b0(xs1))].as_chunks::<4>().0;
    super::latency::cycles_with(rs1_iter(xs1), xs2, |i| f32::from_le_bytes(words[i]))
}
