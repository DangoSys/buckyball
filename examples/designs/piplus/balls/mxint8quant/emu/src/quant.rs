//! MXINT8QUANT: rows of K FP32 values to packed MXINT8 rows (K INT8 codes,
//! then K/32 E8M0 exponent bytes, zero-padded to whole bank rows), the input
//! rows in_pitch bank rows apart. With columns, the input is a row-major
//! matrix [K][rows], its rows in_pitch bank rows apart, and the output rows
//! are its columns.
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use bebop_bemu::ExecContext;

/// Bank rows of one packed row of length k.
fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

/// The block exponent e (stored as e + 127) of 32 FP32 values.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
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

/// Scaling by a power of two preserves the significand whenever a value
/// can round to a nonzero code. Values below that range round to zero.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
fn encode_block(values: &[u32; 32], dest: &mut [u8; 32]) -> u8 {
    let exponent = block_exponent(values);
    let scale = f32::from_bits(((127 - exponent) as u32) << 23);
    for (code, &bits) in dest.iter_mut().zip(values) {
        let scaled = f32::from_bits(bits) * scale;
        *code = scaled.round_ties_even().clamp(-127.0, 127.0) as i8 as u8;
    }
    (exponent + 127) as u8
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
fn encode_block(values: &[u32; 32], dest: &mut [u8; 32]) -> u8 {
    use std::arch::x86_64::*;
    // Two unaligned 16-lane loads cover the input exactly. Saturating before
    // rounding gives the same signed codes at the two integer endpoints.
    unsafe {
        let lo = _mm512_loadu_si512(values.as_ptr().cast());
        let hi = _mm512_loadu_si512(values.as_ptr().add(16).cast());
        let abs = _mm512_set1_epi32(0x7fffffff);
        let max = _mm512_reduce_max_epu32(_mm512_max_epu32(_mm512_and_si512(lo, abs), _mm512_and_si512(hi, abs)));
        let exponent = if max == 0 {
            0
        } else {
            ((max >> 23) as i32 - 133).max(-127)
        };
        let scale = _mm512_set1_ps(f32::from_bits(((127 - exponent) as u32) << 23));
        for (lane, raw) in [(0, lo), (16, hi)] {
            let scaled = _mm512_mul_ps(_mm512_castsi512_ps(raw), scale);
            let clipped = _mm512_max_ps(_mm512_set1_ps(-127.0), _mm512_min_ps(scaled, _mm512_set1_ps(127.0)));
            let codes = _mm512_cvt_roundps_epi32::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(clipped);
            _mm_storeu_si128(dest.as_mut_ptr().add(lane).cast(), _mm512_cvtsepi32_epi8(codes));
        }
        (exponent + 127) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantization_observes_ties_clamps_and_subnormal_blocks() {
        for field in 0..255u32 {
            let values: [u32; 32] = std::array::from_fn(|i| {
                let e = field.saturating_sub((i % 6) as u32);
                let fraction = [0, 0x7fffff, 0x10000, 0x30000, 0x0ffff, 0x10001, 1, 0x400000][i % 8];
                ((i as u32 % 2) << 31) | (e << 23) | fraction
            });
            let mut codes = [0; 32];
            let biased = encode_block(&values, &mut codes);
            let unit = 2f64.powi(i32::from(biased) - 127);
            for (bits, code) in values.into_iter().zip(codes) {
                let expected = (f64::from(f32::from_bits(bits)) / unit)
                    .round_ties_even()
                    .clamp(-127.0, 127.0);
                assert_eq!(code as i8, expected as i8);
            }
        }
    }
}
pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let k = (xs2 & 0xffff) as usize;
    let in_base = ((xs2 >> 16) & 0x3fff) as usize;
    let out_base = ((xs2 >> 30) & 0x3fff) as usize;
    let columns = (xs2 >> 44) & 1 == 1;
    let in_pitch = ((xs2 >> 45) & 0x3fff) as usize;
    assert!(
        rs1_b1(xs1) == 0 && !rs1_read_old(xs1) && xs2 >> 59 == 0,
        "mxint8quant: reserved fields must be zero"
    );
    assert!(
        rows > 0 && k > 0 && k % 32 == 0 && (!columns || rows % 4 == 0),
        "mxint8quant: rows > 0 (a multiple of 4 with columns) and K a positive multiple of 32"
    );
    assert!(input != output, "mxint8quant: input and output banks must differ");
    for bank in [input, output] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "mxint8quant: banks must be allocated with one group"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "mxint8quant: bank rows must be 128 bits");
    let stride = packed_rows(k);
    // The bank rows of an input row and the input rows.
    let (width, count) = if columns { (rows / 4, k) } else { (k / 4, rows) };
    assert!(in_pitch >= width, "mxint8quant: input rows overlap");
    assert!(
        in_base + (count - 1) * in_pitch + width <= bank_lines() && out_base + rows * stride <= bank_lines(),
        "mxint8quant: bank window exceeds capacity"
    );
    let source = pbank(ctx, input);
    let pitch = in_pitch * 16;
    let start = in_base * 16;
    for row in 0..count {
        let at = start + row * pitch;
        assert!(
            ctx.banks[source][at..at + width * 16]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| f32::from_le_bytes(*v).is_finite()),
            "mxint8quant: non-finite input"
        );
    }
    let destination = pbank(ctx, output);
    let (source, destination) = ctx.banks.read_write(source, destination);
    let packed = &mut destination[out_base * 16..(out_base + rows * stride) * 16];
    let source = &source[start..];
    for (row, packed) in packed.chunks_exact_mut(stride * 16).enumerate() {
        let (codes, exponents) = packed.split_at_mut(k);
        for (block, dest) in codes.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let values = std::array::from_fn(|lane| {
                let i = 32 * block + lane;
                let at = if columns {
                    i * pitch + row * 4
                } else {
                    row * pitch + i * 4
                };
                u32::from_le_bytes(source[at..at + 4].try_into().unwrap())
            });
            exponents[block] = encode_block(&values, dest);
        }
        exponents[k / 32..].fill(0);
    }
    0
}

/// The cycles of the RTL (latency.rs).
pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    super::latency::cycles(rs1_iter(xs1), xs2)
}
