// Finite FP16 values are exact signed integers in units of 2^-24.
fn finite(bits: u16) -> i128 {
    let exponent = (bits >> 10) & 31;
    let mantissa = (bits & 1023) as i128;
    let value = if exponent == 0 {
        mantissa
    } else {
        (1024 + mantissa) << (exponent - 1)
    };
    if bits & 0x8000 != 0 {
        -value
    } else {
        value
    }
}

pub(super) fn fma(a: u16, b: u16, c: u16) -> u16 {
    let aa = a & 0x7fff;
    let bb = b & 0x7fff;
    let cc = c & 0x7fff;
    let product_sign = (a ^ b) & 0x8000;
    if aa > 0x7c00
        || bb > 0x7c00
        || cc > 0x7c00
        || (aa == 0x7c00 && bb == 0)
        || (bb == 0x7c00 && aa == 0)
    {
        return 0x7e00;
    }
    if aa == 0x7c00 || bb == 0x7c00 {
        return if cc == 0x7c00 && c & 0x8000 != product_sign {
            0x7e00
        } else {
            product_sign | 0x7c00
        };
    }
    if cc == 0x7c00 {
        return c;
    }
    // The largest product uses fewer than 81 bits; i128 retains every bit.
    let sum = finite(a) * finite(b) + (finite(c) << 24);
    if sum == 0 {
        return if (aa == 0 || bb == 0) && cc == 0 {
            product_sign & c & 0x8000
        } else {
            0
        };
    }
    let sign = if sum < 0 { 0x8000 } else { 0 };
    let magnitude = sum.unsigned_abs();
    let highest = 127 - magnitude.leading_zeros();
    let shift = if highest < 34 { 24 } else { highest - 10 };
    let mut significand = magnitude >> shift;
    let remainder = magnitude & ((1u128 << shift) - 1);
    let halfway = 1u128 << (shift - 1);
    if remainder > halfway || (remainder == halfway && significand & 1 != 0) {
        significand += 1;
    }
    if highest < 34 {
        return sign | significand as u16;
    }
    let mut exponent = highest as i32 - 48 + 15;
    if significand == 2048 {
        significand = 1024;
        exponent += 1;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    sign | (exponent as u16) << 10 | (significand as u16 & 1023)
}
