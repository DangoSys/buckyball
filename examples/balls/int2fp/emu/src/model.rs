pub fn int2fp_fp32_bits(value: i32, scale_bits: u32) -> u32 {
    fp32_multiply(int32_to_fp32(value), scale_bits)
}

pub fn int2fp_i8_bits(value: i32, scale_bits: u32) -> i8 {
    fp32_to_int32(int2fp_fp32_bits(value, scale_bits)).clamp(-128, 127) as i8
}

pub fn int32_to_fp32(value: i32) -> u32 {
    if value == 0 {
        return 0;
    }

    let sign = ((value as u32) >> 31) & 1;
    let abs = if value == i32::MIN {
        0x8000_0000u32
    } else {
        value.unsigned_abs()
    };

    let leading_one = 31u32 - abs.leading_zeros();
    let mut exponent = leading_one + 127;
    let significand = if leading_one > 23 {
        let right_shift = leading_one - 23;
        let abs_wide = u64::from(abs);
        let truncated = abs_wide >> right_shift;
        let half = 1u64 << (right_shift - 1);
        let remainder = abs_wide & ((1u64 << right_shift) - 1);
        let round_up = remainder > half || (remainder == half && (truncated & 1) != 0);
        let rounded = truncated + u64::from(round_up);
        if ((rounded >> 24) & 1) != 0 {
            exponent = leading_one + 128;
            (rounded >> 1) as u32
        } else {
            rounded as u32
        }
    } else {
        abs << (23 - leading_one)
    };

    (sign << 31) | ((exponent & 0xff) << 23) | (significand & 0x7f_ffff)
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
    fn int_to_fp_basic() {
        let scale = 0x3F80_0000;
        assert_eq!(int2fp_fp32_bits(1, scale), 0x3F80_0000);
        assert_eq!(int2fp_fp32_bits(-1, scale), 0xBF80_0000);
        assert_eq!(int2fp_fp32_bits(0, scale), 0);
        assert_eq!(int2fp_fp32_bits(i32::MIN, scale), 0xCF00_0000);
    }

    #[test]
    fn int8_to_fp_scale() {
        let scale = 0x3E80_0000; // 0.25
        assert_eq!(int2fp_fp32_bits(-128, scale), (-32.0f32).to_bits());
        assert_eq!(int2fp_fp32_bits(127, scale), (31.75f32).to_bits());
    }

    #[test]
    fn requant_vectors() {
        let scale = 0x3F00_0000; // 0.5
        let input = [
            -1000, -257, -255, -5, -3, -1, 0, 1, 3, 5, 127, 253, 255, 257, 1000, 2, -999, -511,
            -259, -9, -7, -3, 2, 4, 6, 9, 125, 251, 254, 258, 511, 999,
        ];
        let expected = [
            -128i8, -128, -128, -2, -2, 0, 0, 0, 2, 2, 64, 126, 127, 127, 127, 1, -128, -128, -128,
            -4, -4, -2, 1, 2, 3, 4, 62, 126, 127, 127, 127, 127,
        ];
        for (value, expected) in input.into_iter().zip(expected) {
            assert_eq!(int2fp_i8_bits(value, scale), expected, "input={value}");
        }
    }
}
