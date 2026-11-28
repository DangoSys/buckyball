/// Compute one 16-wide tile from the current PE or scratchpad values.
/// The packed lanes exist only for this invocation.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512f"))]
fn full_tile(
    spad: &[[i8; 16]],
    base: usize,
    stride: usize,
    rhs: &[[i32; 16]; 16],
    out: &mut [[i32; 16]],
    known_bytes: bool,
) -> bool {
    use std::arch::x86_64::*;
    if out.len() < 4 || !known_bytes && !rhs.iter().flatten().all(|&x| i32::from(x as i8) == x) {
        return false;
    }
    // Each dword contains four signed K values at the same output column.
    let packed: [[u32; 16]; 4] = std::array::from_fn(|group| {
        std::array::from_fn(|j| u32::from_le_bytes(std::array::from_fn(|i| rhs[group * 4 + i][j] as u8)))
    });
    // All vector loads and stores cover a full row. Rebase signed activations
    // to unsigned bytes and subtract the product of the same bias and weights.
    unsafe {
        let dot = |mut sum: __m512i, a: __m512i, b: __m512i| {
            std::arch::asm!(
                "vpdpbusd {sum}, {a}, {b}",
                sum = inout(zmm_reg) sum,
                a = in(zmm_reg) a,
                b = in(zmm_reg) b,
                options(pure, nomem, nostack),
            );
            sum
        };
        let weights = packed.map(|g| _mm512_loadu_si512(g.as_ptr().cast()));
        let mut bias = _mm512_setzero_si512();
        for &v in &weights {
            bias = dot(bias, _mm512_set1_epi8(-128), v);
        }
        for (row, dst) in out.iter_mut().enumerate() {
            let acts = spad[base + row * stride].map(|v| v as u8);
            let mut sum = _mm512_sub_epi32(_mm512_loadu_si512(dst.as_ptr().cast()), bias);
            for (words, &w) in acts.as_chunks::<4>().0.iter().zip(&weights) {
                let a = i32::from_le_bytes(*words) ^ 0x80808080u32 as i32;
                sum = dot(sum, _mm512_set1_epi32(a), w);
            }
            _mm512_storeu_si512(dst.as_mut_ptr().cast(), sum);
        }
    }
    true
}

pub(crate) fn multiply(
    spad: &[[i8; 16]],
    base: usize,
    stride: usize,
    transpose: bool,
    rhs: &[[i32; 16]; 16],
    dest: &mut [[i32; 16]],
    k_len: usize,
    _known_bytes: bool,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512f"))]
    if !transpose && k_len == 16 && full_tile(spad, base, stride, rhs, dest, _known_bytes) {
        return;
    }
    for (i, row) in dest.iter_mut().enumerate() {
        let mut sums = *row;
        for (k, weights) in rhs.iter().take(k_len).enumerate() {
            let (r, c) = if transpose { (k, i) } else { (i, k) };
            let x = i32::from(spad[base + stride * r][c]);
            for (sum, &w) in sums.iter_mut().zip(weights) {
                *sum = sum.wrapping_add(x.wrapping_mul(w));
            }
        }
        *row = sums;
    }
}

#[cfg(test)]
mod tests {
    use super::multiply;
    #[test]
    fn tiles_match_signed_wrapping_multiply_for_all_layouts() {
        let mut seed = 0xf85237a1u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as i32
        };
        let a: Vec<[i8; 16]> = (0..64).map(|_| std::array::from_fn(|_| next() as i8)).collect();
        let bytes: [[i32; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| next() as i8 as i32));
        let mut wide = bytes;
        wide[3][2] = i32::MIN;
        wide[9][12] = i32::MAX;
        let initial: [[i32; 16]; 16] = std::array::from_fn(|_| std::array::from_fn(|_| next()));
        for (rhs, known_bytes) in [(&bytes, true), (&bytes, false), (&wide, false)] {
            for stride in [0, 1, 3] {
                for transpose in [false, true] {
                    for (rows, k_len) in [(1, 1), (3, 16), (4, 16), (13, 9), (16, 16)] {
                        let mut actual = initial;
                        let mut expected = initial;
                        multiply(&a, 2, stride, transpose, rhs, &mut actual[..rows], k_len, known_bytes);
                        for i in 0..rows {
                            for j in 0..16 {
                                for k in 0..k_len {
                                    let x = if transpose {
                                        a[2 + stride * k][i]
                                    } else {
                                        a[2 + stride * i][k]
                                    };
                                    expected[i][j] = expected[i][j].wrapping_add(i32::from(x).wrapping_mul(rhs[k][j]));
                                }
                            }
                        }
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}
