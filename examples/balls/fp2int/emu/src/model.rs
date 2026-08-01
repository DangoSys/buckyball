pub fn fp2int_i32_bits(fp_bits: u32, scale_bits: u32) -> i32 {
    fp32_to_int32(fp32_multiply(fp_bits, scale_bits))
}

pub fn fp2int_i8_bits(fp_bits: u32, scale_bits: u32) -> i8 {
    fp2int_i32_bits(fp_bits, scale_bits).clamp(-128, 127) as i8
}

#[allow(dead_code)]
pub fn fp2int_i32_word(input: [u32; 4], scale_bits: u32) -> [i32; 4] {
    [
        fp2int_i32_bits(input[0], scale_bits),
        fp2int_i32_bits(input[1], scale_bits),
        fp2int_i32_bits(input[2], scale_bits),
        fp2int_i32_bits(input[3], scale_bits),
    ]
}

#[allow(dead_code)]
pub fn fp2int_i8_group(input: [u32; 4], scale_bits: u32) -> [i8; 4] {
    [
        fp2int_i8_bits(input[0], scale_bits),
        fp2int_i8_bits(input[1], scale_bits),
        fp2int_i8_bits(input[2], scale_bits),
        fp2int_i8_bits(input[3], scale_bits),
    ]
}

fn fp32_multiply(a: u32, b: u32) -> u32 {
    let a_sign = (a >> 31) & 1;
    let b_sign = (b >> 31) & 1;
    let a_exp = (a >> 23) & 0xff;
    let b_exp = (b >> 23) & 0xff;
    let a_frac = a & 0x7f_ffff;
    let b_frac = b & 0x7f_ffff;
    let a_mant = (1u64 << 23) | u64::from(a_frac);
    let b_mant = (1u64 << 23) | u64::from(b_frac);
    let a_zero = a_exp == 0 && a_frac == 0;
    let b_zero = b_exp == 0 && b_frac == 0;
    let prod = a_mant * b_mant;
    let (sig, guard, round, sticky, exp_adjust) = if ((prod >> 47) & 1) != 0 {
        (
            (prod >> 24) as u32,
            ((prod >> 23) & 1) != 0,
            ((prod >> 22) & 1) != 0,
            (prod & ((1u64 << 22) - 1)) != 0,
            1u32,
        )
    } else {
        (
            (prod >> 23) as u32,
            ((prod >> 22) & 1) != 0,
            ((prod >> 21) & 1) != 0,
            (prod & ((1u64 << 21) - 1)) != 0,
            0u32,
        )
    };
    // Match the RTL's round-to-nearest, ties-to-even multiplier.
    let increment = guard && (round || sticky || (sig & 1) != 0);
    let rounded = u64::from(sig) + u64::from(increment);
    let carry = ((rounded >> 24) & 1) as u32;
    let final_sig = if carry != 0 { rounded >> 1 } else { rounded } as u32;
    let exp_wide = (a_exp + b_exp + exp_adjust + carry).wrapping_sub(127) & 0x3ff;

    if a_zero || b_zero {
        0
    } else if (exp_wide & 0x200) != 0 {
        0
    } else if (exp_wide & 0x100) != 0 {
        ((a_sign ^ b_sign) << 31) | (0xff << 23)
    } else {
        ((a_sign ^ b_sign) << 31) | ((exp_wide & 0xff) << 23) | (final_sig & 0x7f_ffff)
    }
}

fn fp32_to_int32(fp: u32) -> i32 {
    let sign = ((fp >> 31) & 1) != 0;
    let exponent = ((fp >> 23) & 0xff) as i32;
    let frac = fp & 0x7f_ffff;
    let mantissa = (1u32 << 23) | frac;
    let is_zero = exponent == 0 && frac == 0;
    let exp_val = exponent - 127;

    if exponent == 0xff && frac != 0 {
        return 0;
    }
    if is_zero {
        return 0;
    }

    let magnitude = if exp_val >= 31 {
        0x8000_0000u64
    } else if exp_val >= 23 {
        u64::from(mantissa) << (exp_val - 23)
    } else if exp_val >= -1 {
        let right_shift = (23 - exp_val) as u32;
        let truncated = u64::from(mantissa) >> right_shift;
        let half = 1u64 << (right_shift - 1);
        let remainder = u64::from(mantissa) & ((1u64 << right_shift) - 1);
        let round_up = remainder > half || (remainder == half && (truncated & 1) != 0);
        truncated + u64::from(round_up)
    } else {
        0
    };

    if sign {
        if magnitude >= 0x8000_0000 {
            i32::MIN
        } else {
            -(magnitude as i32)
        }
    } else if magnitude > i32::MAX as u64 {
        i32::MAX
    } else {
        magnitude as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int32_basic() {
        let scale = 0x3F80_0000;

        assert_eq!(fp2int_i32_bits(0x3F80_0000, scale), 1);
        assert_eq!(fp2int_i32_bits(0x4000_0000, scale), 2);
        assert_eq!(fp2int_i32_bits(0xBF80_0000, scale), -1);
        assert_eq!(fp2int_i32_bits(0x3F00_0000, scale), 0); //  0.5 -> even 0
        assert_eq!(fp2int_i32_bits(0xBF00_0000, scale), 0); // -0.5 -> even 0
        assert_eq!(fp2int_i32_bits(0x3FC0_0000, scale), 2); //  1.5 -> even 2
        assert_eq!(fp2int_i32_bits(0xBFC0_0000, scale), -2); // -1.5 -> even -2
        assert_eq!(fp2int_i32_bits(0x4020_0000, scale), 2); //  2.5 -> even 2
        assert_eq!(fp2int_i32_bits(0xC020_0000, scale), -2); // -2.5 -> even -2
    }

    #[test]
    fn int8_saturates() {
        let scale = 0x3F80_0000;

        assert_eq!(fp2int_i8_bits(0x4300_0000, scale), 127);
        assert_eq!(fp2int_i8_bits(0xC300_0000, scale), -128);
    }

    #[test]
    fn fp32_to_int8_workload_vectors() {
        let scale = 2.0f32.to_bits();
        let input = [
            0.125f32, -0.125, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 63.25, 63.75,
            -63.75, -64.75, 0.0, -0.0, 2.25, -2.25, 2.75, -2.75, 3.25, -3.25, 3.75, -3.75, 10.125,
            -10.125, 20.25, -20.25, 0.375, -0.375, 64.25, -65.25,
        ];
        let expected = [
            0i8, 0, 0, 0, 2, -2, 2, -2, 4, -4, 126, 127, -128, -128, 0, 0, 4, -4, 6, -6, 6, -6, 8,
            -8, 20, -20, 40, -40, 1, -1, 127, -128,
        ];

        for (value, expected) in input.into_iter().zip(expected) {
            assert_eq!(
                fp2int_i8_bits(value.to_bits(), scale),
                expected,
                "input={value}"
            );
        }
    }
}
