use bebop_bemu::ball::blink::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};

fn divide(value: i128, divisor: u64) -> i64 {
    let magnitude = value.unsigned_abs();
    let mut quotient = magnitude / u128::from(divisor);
    let remainder = magnitude % u128::from(divisor);
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

fn fixed(bits: u32) -> i64 {
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
        divide(mantissa, 1u64 << -shift)
    };
    let value = if bits >> 31 == 0 { magnitude } else { -magnitude };
    i32::try_from(value).expect("INTNORM input must fit signed Q16");
    value
}

pub(super) fn execute(xs1: u64, xs2: u64, ctx: &mut bebop_bemu::ExecContext) -> u64 {
    assert_eq!(xs2 >> 58, 0, "INTNORM reserved fields must be zero");
    let input = rs1_b0(xs1);
    let coefficients = rs1_b1(xs1);
    let output = rs1_b2(xs1);
    assert!(
        input != coefficients && input != output && coefficients != output,
        "INTNORM banks must differ"
    );
    for bank in [input, coefficients, output] {
        let config = ctx.config(bank);
        assert!(
            config.allocated && config.cols == 1,
            "INTNORM requires allocated one-group banks"
        );
    }
    let rows = rs1_iter(xs1) as usize;
    let k = (xs2 & 65535) as usize;
    assert!(
        rows > 0 && k > 0 && k % 512 == 0,
        "INTNORM requires rows and a multiple of 512 elements"
    );
    let input_base = ((xs2 >> 16) & 16383) as usize * 16;
    let output_base = ((xs2 >> 30) & 16383) as usize * 16;
    let coefficient_base = ((xs2 >> 44) & 16383) as usize * 16;
    let stride = k + k / 32;
    let scratch = output_base + rows * stride;
    let a = pbank_group(ctx, input, 0);
    let b = pbank_group(ctx, coefficients, 0);
    let c = pbank_group(ctx, output, 0);
    assert!(
        input_base + rows * k * 4 <= ctx.banks[a].len(),
        "INTNORM input span exceeds bank"
    );
    assert!(
        coefficient_base + 16 + k * 8 <= ctx.banks[b].len(),
        "INTNORM coefficient span exceeds bank"
    );
    assert!(
        scratch + k * 4 <= ctx.banks[c].len(),
        "INTNORM output and scratch exceed bank"
    );
    let epsilon = u64::from_le_bytes(ctx.banks[b][coefficient_base..coefficient_base + 8].try_into().unwrap());
    assert!(epsilon > 0, "INTNORM epsilon must be positive");
    assert!(
        ctx.banks[b][coefficient_base + 8..coefficient_base + 16]
            .iter()
            .all(|v| *v == 0),
        "INTNORM reserved coefficient words must be zero"
    );
    let mut gamma = Vec::with_capacity(k);
    let mut beta = Vec::with_capacity(k);
    for i in 0..k {
        let offset = coefficient_base + 16 + i * 4;
        gamma.push(i32::from_le_bytes(ctx.banks[b][offset..offset + 4].try_into().unwrap()));
        beta.push(i32::from_le_bytes(
            ctx.banks[b][offset + k * 4..offset + k * 4 + 4].try_into().unwrap(),
        ));
    }
    for row in 0..rows {
        let values: Vec<i64> = (0..k)
            .map(|i| {
                let offset = input_base + (row * k + i) * 4;
                fixed(u32::from_le_bytes(ctx.banks[a][offset..offset + 4].try_into().unwrap()))
            })
            .collect();
        let mean = divide(values.iter().map(|v| i128::from(*v)).sum(), k as u64);
        let squares: i128 = values.iter().map(|v| i128::from(*v - mean).pow(2)).sum();
        let variance = u64::try_from(divide(squares, k as u64))
            .unwrap()
            .checked_add(epsilon)
            .expect("INTNORM variance exceeds u64");
        let root = variance.isqrt();
        let mut normalized = Vec::with_capacity(k);
        for i in 0..k {
            let z = divide(i128::from(values[i] - mean) * 65536, root);
            let value = divide(i128::from(z) * i128::from(gamma[i]), 65536) + i64::from(beta[i]);
            let value = i32::try_from(value).expect("INTNORM result must fit signed Q16");
            ctx.banks[c][scratch + i * 4..scratch + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            normalized.push(value);
        }
        let target = output_base + row * stride;
        for (block, values) in normalized.chunks_exact(32).enumerate() {
            let maximum = values.iter().map(|v| v.unsigned_abs()).max().unwrap();
            let bits = u32::BITS - maximum.leading_zeros();
            ctx.banks[c][target + k + block] = if maximum == 0 { 127 } else { (bits + 104) as u8 };
            for (i, value) in values.iter().enumerate() {
                let code = if bits >= 7 {
                    divide(i128::from(*value), 1u64 << (bits - 7))
                } else {
                    i64::from(*value) * (1i64 << (7 - bits))
                };
                ctx.banks[c][target + block * 32 + i] = code.clamp(-127, 127) as i8 as u8;
            }
        }
    }
    0
}
