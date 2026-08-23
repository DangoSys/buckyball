pub fn fp2int_i32_bits(fp_bits: u32, scale_bits: u32) -> i32 {
    fp32_to_int32(fp32_multiply(fp_bits, scale_bits))
}

pub fn fp2int_i8_bits(fp_bits: u32, scale_bits: u32) -> i8 {
    fp2int_i32_bits(fp_bits, scale_bits).clamp(-128, 127) as i8
}

pub fn fp2int_da_bits(input_words: &[u128]) -> u32 {
    let mut max_abs = 0u32;
    for word in input_words {
        for lane in 0..4 {
            let bits = (*word >> (lane * 32)) as u32;
            assert_ne!(
                (bits >> 23) & 0xff,
                0xff,
                "Fp2Int does not accept NaN or infinity"
            );
            max_abs = max_abs.max(bits & 0x7fff_ffff);
        }
    }
    fp2int_da_from_max_abs_bits(max_abs)
}

pub fn fp2int_da_from_max_abs_bits(max_abs: u32) -> u32 {
    if max_abs == 0 {
        0x3f80_0000
    } else {
        fp32_divide(max_abs, 0x42fe_0000)
    }
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

pub fn fp32_divide(a: u32, b: u32) -> u32 {
    let a_exp = (a >> 23) & 0xff;
    let b_exp = (b >> 23) & 0xff;
    let a_mant = (1u64 << 23) | u64::from(a & 0x7f_ffff);
    let b_mant = (1u64 << 23) | u64::from(b & 0x7f_ffff);
    let normalize_down = a_mant < b_mant;
    let dividend = if normalize_down {
        a_mant << 26
    } else {
        a_mant << 25
    };
    let quotient = dividend / b_mant;
    let remainder = dividend % b_mant;
    let sig = quotient >> 2;
    let round_up = (quotient & 0b10) != 0
        && ((quotient & 0b1) != 0 || remainder != 0 || (sig & 0b1) != 0);
    let rounded = sig + u64::from(round_up);
    let exp = a_exp as i32 - b_exp as i32
        + if normalize_down { 126 } else { 127 }
        + ((rounded >> 24) as i32);

    if (a & 0x7fff_ffff) == 0 {
        0
    } else if (b & 0x7fff_ffff) == 0 || exp > 254 {
        ((a ^ b) & 0x8000_0000) | 0x7f80_0000
    } else if exp < 1 {
        0
    } else {
        ((a ^ b) & 0x8000_0000)
            | ((exp as u32) << 23)
            | if (rounded & (1 << 24)) != 0 {
                0
            } else {
                (rounded as u32) & 0x7f_ffff
            }
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
