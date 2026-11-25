// Reference of INTNORM ported from the BEMU model (emu/src/norm.rs). Banks
// are byte images of one group; bases and footprints are in bytes.

// Events the coverage asks for, counted while the reference runs.
#[derive(Default)]
pub struct Stats {
    pub input_ties: u32,
    pub code_ties: u32,
    pub clips: u32,
    pub zero_blocks: u32,
    pub narrow_blocks: u32,
    pub full_blocks: u32,
    pub negative_means: u32,
    pub epsilon_rows: u32,
}

// Signed division rounded to nearest, ties to even on the magnitude.
fn divide(value: i128, divisor: u64, ties: &mut u32) -> i64 {
    let magnitude = value.unsigned_abs();
    let mut quotient = magnitude / u128::from(divisor);
    let remainder = magnitude % u128::from(divisor);
    if remainder * 2 == u128::from(divisor) {
        *ties += 1;
    }
    if remainder * 2 > u128::from(divisor) || (remainder * 2 == u128::from(divisor) && quotient & 1 != 0) {
        quotient += 1;
    }
    let rounded = i64::try_from(quotient).unwrap();
    if value < 0 {
        -rounded
    } else {
        rounded
    }
}

// FP32 bit pattern to signed Q16 with RNE.
pub fn fixed(bits: u32, ties: &mut u32) -> i64 {
    let exponent = (bits >> 23) & 255;
    assert!(exponent < 255, "INTNORM requires finite input");
    if exponent == 0 {
        return 0;
    }
    assert!(exponent <= 142, "INTNORM input must fit signed Q16");
    let mantissa = i128::from((bits & 0x7fffff) | 0x800000);
    let shift = exponent as i32 - 134;
    let magnitude = if shift >= 0 {
        (mantissa << shift) as i64
    } else if shift <= -25 {
        0
    } else {
        divide(mantissa, 1u64 << -shift, ties)
    };
    let value = if bits >> 31 == 0 { magnitude } else { -magnitude };
    i32::try_from(value).expect("INTNORM input must fit signed Q16");
    value
}

fn word(bank: &[u8], offset: usize) -> [u8; 4] {
    bank[offset..offset + 4].try_into().unwrap()
}

// Runs one command over the byte images. Returns the Q16 results of every
// row, which the Ball keeps in the scratch span of the output bank.
#[allow(clippy::too_many_arguments)]
pub fn execute(
    input: &[u8],
    coefficients: &[u8],
    output: &mut [u8],
    rows: usize,
    k: usize,
    input_base: usize,
    coefficient_base: usize,
    output_base: usize,
    stats: &mut Stats,
) -> Vec<Vec<i32>> {
    assert!(rows > 0 && k > 0 && k % 512 == 0);
    let stride = k + k / 32;
    let scratch = output_base + rows * stride;
    assert!(input_base + rows * k * 4 <= input.len());
    assert!(coefficient_base + 16 + k * 8 <= coefficients.len());
    assert!(scratch + k * 4 <= output.len());
    let epsilon = u64::from_le_bytes(coefficients[coefficient_base..coefficient_base + 8].try_into().unwrap());
    assert!(epsilon > 0, "INTNORM epsilon must be positive");
    assert!(coefficients[coefficient_base + 8..coefficient_base + 16].iter().all(|v| *v == 0));
    let gamma: Vec<i32> = (0..k)
        .map(|i| i32::from_le_bytes(word(coefficients, coefficient_base + 16 + i * 4)))
        .collect();
    let beta: Vec<i32> = (0..k)
        .map(|i| i32::from_le_bytes(word(coefficients, coefficient_base + 16 + (k + i) * 4)))
        .collect();
    let mut results = Vec::with_capacity(rows);
    let mut unused = 0;
    for row in 0..rows {
        let values: Vec<i64> = (0..k)
            .map(|i| fixed(u32::from_le_bytes(word(input, input_base + (row * k + i) * 4)), &mut stats.input_ties))
            .collect();
        let mean = divide(values.iter().map(|v| i128::from(*v)).sum(), k as u64, &mut unused);
        if mean < 0 {
            stats.negative_means += 1;
        }
        let squares: i128 = values.iter().map(|v| i128::from(*v - mean).pow(2)).sum();
        let deviation = u64::try_from(divide(squares, k as u64, &mut unused)).unwrap();
        if epsilon > deviation {
            stats.epsilon_rows += 1;
        }
        let variance = deviation.checked_add(epsilon).expect("INTNORM variance exceeds u64");
        let root = variance.isqrt();
        let mut normalized = Vec::with_capacity(k);
        for i in 0..k {
            let z = divide(i128::from(values[i] - mean) * 65536, root, &mut unused);
            let value = divide(i128::from(z) * i128::from(gamma[i]), 65536, &mut unused) + i64::from(beta[i]);
            let value = i32::try_from(value).expect("INTNORM result must fit signed Q16");
            output[scratch + i * 4..scratch + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            normalized.push(value);
        }
        let target = output_base + row * stride;
        for (block, values) in normalized.chunks_exact(32).enumerate() {
            let maximum = values.iter().map(|v| v.unsigned_abs()).max().unwrap();
            let bits = u32::BITS - maximum.leading_zeros();
            output[target + k + block] = if maximum == 0 { 127 } else { (bits + 104) as u8 };
            if maximum == 0 {
                stats.zero_blocks += 1;
            } else if bits < 7 {
                stats.narrow_blocks += 1;
            } else if bits == 32 {
                stats.full_blocks += 1;
            }
            for (i, value) in values.iter().enumerate() {
                let code = if bits >= 7 {
                    divide(i128::from(*value), 1u64 << (bits - 7), &mut stats.code_ties)
                } else {
                    i64::from(*value) * (1i64 << (7 - bits))
                };
                if code.abs() > 127 {
                    stats.clips += 1;
                }
                output[target + block * 32 + i] = code.clamp(-127, 127) as i8 as u8;
            }
        }
        results.push(normalized);
    }
    results
}
