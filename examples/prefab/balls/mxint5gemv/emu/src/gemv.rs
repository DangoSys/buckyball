use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use bebop_bemu::ball::blink::{rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use bebop_bemu::ExecContext;

fn round_shift(value: i64, shift: u32) -> i64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }
    let magnitude = value.unsigned_abs();
    let quotient = magnitude >> shift;
    let remainder = magnitude & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let rounded = quotient + u64::from(remainder > halfway || remainder == halfway && quotient & 1 != 0);
    if value < 0 {
        -(rounded as i64)
    } else {
        rounded as i64
    }
}

// Four dot products use the same 32 activation bytes. Horizontal additions
// combine four columns together, keeping independent accumulator chains.
#[inline]
fn dot5_four(x: &[u8; 32], rows: [&[u8; 20]; 4]) -> [i32; 4] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi",
        target_feature = "avx512vl",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
        target_feature = "avx2"
    ))]
    {
        use std::arch::x86_64::*;
        // All loads are bounded by the input arrays. Only the low 20 bytes of
        // each weight vector are active in the masked loads.
        unsafe {
            let x = _mm256_loadu_si256(x.as_ptr().cast());
            let magnitude = _mm256_abs_epi8(x);
            let indices = _mm256_setr_epi64x(0x0403020100, 0x0908070605, 0x0e0d0c0b0a, 0x131211100f);
            let shifts = _mm256_set1_epi64x(0x231e19140f0a0500);
            let sign = _mm256_set1_epi8(16);
            let mut invalid = _mm256_setzero_si256();
            let sums = rows.map(|row| {
                let packed = _mm256_maskz_loadu_epi8(0xfffff, row.as_ptr().cast());
                let groups = _mm256_permutexvar_epi8(indices, packed);
                let fields = _mm256_and_si256(_mm256_multishift_epi64_epi8(shifts, groups), _mm256_set1_epi8(31));
                invalid = _mm256_or_si256(invalid, _mm256_cmpeq_epi8(fields, sign));
                let weights = _mm256_sub_epi8(_mm256_xor_si256(fields, sign), sign);
                let signed = _mm256_sign_epi8(weights, x);
                let mut result = _mm256_setzero_si256();
                std::arch::asm!("vpdpbusd {acc}, {x}, {w}", acc=inout(ymm_reg) result,
                    x=in(ymm_reg) magnitude, w=in(ymm_reg) signed, options(pure,nomem,nostack));
                result
            });
            assert_eq!(_mm256_movemask_epi8(invalid), 0, "mxint5gemv: reserved weight code");
            let pairs01 = _mm256_hadd_epi32(sums[0], sums[1]);
            let pairs23 = _mm256_hadd_epi32(sums[2], sums[3]);
            let halves = _mm256_hadd_epi32(pairs01, pairs23);
            let result = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256::<1>(halves));
            let mut values = [0; 4];
            _mm_storeu_si128(values.as_mut_ptr().cast(), result);
            values
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi",
        target_feature = "avx512vl",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
        target_feature = "avx2"
    )))]
    {
        rows.map(|packed| {
            let mut dot = 0;
            for (bytes, xs) in packed.chunks_exact(5).zip(x.chunks_exact(8)) {
                let bits = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], 0, 0, 0]);
                for (lane, &x) in xs.iter().enumerate() {
                    let w = (((bits >> (lane * 5)) & 31) as i32 ^ 16) - 16;
                    assert_ne!(w, -16, "mxint5gemv: reserved weight code");
                    dot += i32::from(x as i8) * w;
                }
            }
            dot
        })
    }
}

#[cfg(test)]
mod tests {
    use super::dot5_four;

    #[test]
    fn packed_dot_preserves_signed_codes_and_rejects_the_reserved_code() {
        let xs: [u8; 32] = std::array::from_fn(|i| [128, 127, 0, 250, 31][i % 5]);
        for rotation in 0..31 {
            let mut bytes = [0u8; 20];
            let mut expected = 0;
            for (i, &x) in xs.iter().enumerate() {
                let w = ((i + rotation) % 31) as i32 - 15;
                expected += i32::from(x as i8) * w;
                for bit in 0..5 {
                    bytes[(5 * i + bit) / 8] |= (((w as u8 >> bit) & 1) << ((5 * i + bit) % 8)) as u8;
                }
            }
            let zero = [0; 20];
            for slot in 0..4 {
                let mut rows = [&zero; 4];
                rows[slot] = &bytes;
                let mut result = [0; 4];
                result[slot] = expected;
                assert_eq!(dot5_four(&xs, rows), result);
            }
        }
        let mut invalid = [0; 20];
        invalid[19] = 0x80; // -16 in the last lane
        assert!(std::panic::catch_unwind(|| dot5_four(&xs, [&invalid; 4])).is_err());
    }
}

pub(crate) fn execute(rs1: u64, rs2: u64, ctx: &mut ExecContext) -> u64 {
    let (a, b, c) = (rs1_b0(rs1), rs1_b1(rs1), rs1_b2(rs1));
    let k = rs1_iter(rs1) as usize;
    let a_codes = (rs2 & 0x3fff) as usize;
    let a_exponents = ((rs2 >> 14) & 0x3fff) as usize;
    let b_base = ((rs2 >> 28) & 0x3fff) as usize;
    let c_base = ((rs2 >> 42) & 0x3fff) as usize;
    let accumulate = rs2 & (1 << 56) != 0;
    let finalize = rs2 & (1 << 57) != 0;
    assert!(rs2 >> 58 == 0, "mxint5gemv: reserved fields must be zero");
    assert_eq!(
        rs1_read_old(rs1),
        accumulate,
        "mxint5gemv: inconsistent accumulate fields"
    );
    assert!(k > 0 && k % 32 == 0, "mxint5gemv: K must be positive and aligned to 32");
    assert!(a != b && a != c && b != c, "mxint5gemv: banks must differ");
    for (bank, groups) in [(a, 1), (b, 4), (c, 1)] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == groups,
            "mxint5gemv: invalid bank group allocation"
        );
    }
    assert_eq!(bank_row_bytes(), 16, "mxint5gemv: bank rows must be 128 bits");
    assert_eq!(bank_lines(), 1024, "mxint5gemv: banks must be 16 KiB");
    let blocks = k / 32;
    let code_rows = (blocks * 5).div_ceil(4);
    let metadata_rows = blocks.div_ceil(16);
    let stride = code_rows + 2 * metadata_rows;
    assert!(
        a_codes + k / 16 <= bank_lines()
            && a_exponents + blocks.div_ceil(16) <= bank_lines()
            && b_base + 8 * stride <= bank_lines()
            && c_base + 17 <= bank_lines(),
        "mxint5gemv: bank window exceeds capacity"
    );
    let pa = ctx.physical_bank(a, 0);
    let pc = ctx.physical_bank(c, 0);
    let codes = &ctx.banks[pa][a_codes * 16..a_codes * 16 + k];
    let exponents = &ctx.banks[pa][a_exponents * 16..a_exponents * 16 + blocks];
    let weights = [0, 1, 2, 3].map(|group| {
        let physical = ctx.physical_bank(b, group);
        &ctx.banks[physical][b_base * 16..(b_base + 8 * stride) * 16]
    });
    let mut values = [0i64; 32];
    let mut scale = 0i16;
    let mut valid = false;
    let mut processed = 0usize;
    if accumulate {
        let state = &ctx.banks[pc][c_base * 16..c_base * 16 + 272];
        processed = u16::from_le_bytes(state[0..2].try_into().unwrap()) as usize;
        scale = i16::from_le_bytes(state[2..4].try_into().unwrap());
        assert!(
            state[4] <= 1 && state[5] == 5 && state[6] == 28 && state[7..16].iter().all(|x| *x == 0),
            "mxint5gemv: invalid state header"
        );
        valid = state[4] != 0;
        for (j, value) in values.iter_mut().enumerate() {
            *value = i64::from_le_bytes(state[16 + j * 8..24 + j * 8].try_into().unwrap());
        }
        assert!(
            processed > 0 && processed <= 1024,
            "mxint5gemv: invalid state block count"
        );
        assert!(
            if valid {
                (-258..=250).contains(&scale) && values.iter().any(|x| *x != 0)
            } else {
                scale == 0 && values.iter().all(|x| *x == 0)
            },
            "mxint5gemv: invalid state scale"
        );
        assert!(
            values.iter().all(|x| x.unsigned_abs() <= (processed as u64) << 52),
            "mxint5gemv: invalid state magnitude"
        );
    }
    assert!(processed + blocks <= 1024, "mxint5gemv: total K exceeds 32768");
    for block in 0..blocks {
        let mut products = [0i64; 32];
        let mut scales = [0i16; 32];
        assert!(exponents[block] != 255, "mxint5gemv: reserved A exponent");
        let x = codes[block * 32..][..32].try_into().unwrap();
        for (group, weight) in weights.iter().enumerate() {
            for first in [0, 4] {
                let rows: [&[u8]; 4] = std::array::from_fn(|lane| {
                    let offset = (first + lane) * stride * 16;
                    &weight[offset..offset + stride * 16]
                });
                let dots = dot5_four(x, rows.map(|row| row[block * 20..][..20].try_into().unwrap()));
                for (lane, (row, dot)) in rows.into_iter().zip(dots).enumerate() {
                    let j = group * 8 + first + lane;
                    let factor = row[code_rows * 16 + block];
                    let exponent = row[(code_rows + metadata_rows) * 16 + block];
                    assert!((1..=137).contains(&factor), "mxint5gemv: invalid integer factor");
                    assert!(exponent != 255, "mxint5gemv: reserved B exponent");
                    scales[j] = exponents[block] as i16 + exponent as i16 - 258;
                    products[j] = i64::from(dot) * i64::from(factor);
                }
            }
        }
        if let Some(incoming) = (0..32).filter(|j| products[*j] != 0).map(|j| scales[j]).max() {
            let next = if valid { scale.max(incoming) } else { incoming };
            if valid && next > scale {
                for value in &mut values {
                    *value = round_shift(*value, (next - scale) as u32);
                }
            }
            for j in 0..32 {
                if products[j] == 0 {
                    continue;
                }
                let shift = 28 + i32::from(scales[j]) - i32::from(next);
                let term = if shift >= 0 {
                    products[j] << shift
                } else {
                    round_shift(products[j], (-shift) as u32)
                };
                values[j] = values[j].checked_add(term).expect("mxint5gemv: INT64 overflow");
            }
            valid = values.iter().any(|x| *x != 0);
            scale = if valid { next } else { 0 };
        }
    }
    processed += blocks;
    if finalize {
        let maximum = values.iter().map(|x| x.unsigned_abs()).max().unwrap();
        let exponent = if maximum == 0 {
            0
        } else {
            (63 - maximum.leading_zeros() as i32 + i32::from(scale) - 28 - 6).clamp(-127, 127)
        };
        let shift = exponent - i32::from(scale) + 28;
        let mut output = [0u8; 48];
        for j in 0..32 {
            let code = if shift >= 0 {
                round_shift(values[j], shift as u32)
            } else if shift < -6 || values[j].unsigned_abs() > (127u64 >> -shift) {
                values[j].signum() * 127
            } else {
                values[j] << -shift
            };
            output[j] = code.clamp(-127, 127) as i8 as u8;
        }
        output[32] = (exponent + 127) as u8;
        ctx.banks[pc][c_base * 16..c_base * 16 + 48].copy_from_slice(&output);
    } else {
        let mut output = [0u8; 272];
        output[0..2].copy_from_slice(&(processed as u16).to_le_bytes());
        output[2..4].copy_from_slice(&scale.to_le_bytes());
        output[4] = u8::from(valid);
        output[5] = 5;
        output[6] = 28;
        for (j, value) in values.iter().enumerate() {
            output[16 + j * 8..24 + j * 8].copy_from_slice(&value.to_le_bytes());
        }
        ctx.banks[pc][c_base * 16..c_base * 16 + 272].copy_from_slice(&output);
    }
    if bebop_bemu::ball::blink::mtrace_enabled() {
        for row in 0..if finalize { 3 } else { 17 } {
            let bytes = &ctx.banks[pc][(c_base + row) * 16..(c_base + row + 1) * 16];
            bebop_bemu::ball::blink::mtrace(bebop_bemu::ball::blink::MTraceEvent {
                is_write: true,
                is_shared: bebop_bemu::ball::blink::is_shared_vbank(c),
                channel: 0,
                hart_id: ctx.hart_id as u64,
                rob_id: ctx.inst_id as u32,
                vbank_id: c as u32,
                pbank_id: ctx.reported_physical_bank(c, pc),
                group_id: 0,
                addr: (c_base + row) as u32,
                write_mask: 0xffff,
                data_lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                data_hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            });
        }
    }
    0
}

// The cycles of the Ball against an ideal outside world (latency.rs), from the
// A and B banks the instruction reads.
pub(crate) fn cycles(rs1: u64, rs2: u64, ctx: &ExecContext) -> u64 {
    let k = rs1_iter(rs1);
    assert!(k > 0 && k % 32 == 0, "mxint5gemv: invalid K");
    let a = &ctx.banks[ctx.physical_bank(rs1_b0(rs1), 0)];
    let b = [0u64, 1, 2, 3].map(|group| &ctx.banks[ctx.physical_bank(rs1_b1(rs1), group)][..]);
    super::latency::cycles(k, rs2, a, b)
}
