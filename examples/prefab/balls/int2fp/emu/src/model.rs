pub fn int2fp_fp32_bits(value: i32, dw_bits: u32) -> u32 {
    fp32_multiply(int32_to_fp32(value), dw_bits)
}

pub fn int2fp_dequant_bits(value: i32, da_bits: u32, dw_bits: u32) -> u32 {
    int2fp_fp32_bits(value, fp32_multiply(da_bits, dw_bits))
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
