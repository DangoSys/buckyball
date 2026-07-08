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
  let (mant, exp_adjust) = if ((prod >> 47) & 1) != 0 {
    ((prod >> 24) as u32, 1u32)
  } else {
    ((prod >> 23) as u32, 0u32)
  };
  let exp_wide = (a_exp + b_exp + exp_adjust).wrapping_sub(127) & 0x3ff;

  if a_zero || b_zero {
    0
  } else if (exp_wide & 0x200) != 0 {
    0
  } else if (exp_wide & 0x100) != 0 {
    ((a_sign ^ b_sign) << 31) | (0xff << 23)
  } else {
    ((a_sign ^ b_sign) << 31) | ((exp_wide & 0xff) << 23) | (mant & 0x7f_ffff)
  }
}

fn fp32_to_int32(fp: u32) -> i32 {
  let sign = ((fp >> 31) & 1) != 0;
  let exponent = ((fp >> 23) & 0xff) as i32;
  let frac = fp & 0x7f_ffff;
  let mantissa = (1u32 << 23) | frac;
  let is_zero = exponent == 0 && frac == 0;
  let exp_val = exponent - 127;

  if is_zero {
    0
  } else if exp_val >= 31 {
    if sign { i32::MIN } else { i32::MAX }
  } else if exp_val < 0 {
    if exp_val == -1 {
      if sign { -1 } else { 1 }
    } else {
      0
    }
  } else {
    let shift = exp_val as u32;
    let mag = if shift >= 23 {
      mantissa << (shift - 23)
    } else {
      mantissa >> (23 - shift)
    };

    if sign { -(mag as i32) } else { mag as i32 }
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
    assert_eq!(fp2int_i32_bits(0x3F00_0000, scale), 1);
    assert_eq!(fp2int_i32_bits(0x3FC0_0000, scale), 1);
    assert_eq!(fp2int_i32_bits(0xC020_0000, scale), -2);
  }

  #[test]
  fn int8_saturates() {
    let scale = 0x3F80_0000;

    assert_eq!(fp2int_i8_bits(0x4300_0000, scale), 127);
    assert_eq!(fp2int_i8_bits(0xC300_0000, scale), -128);
  }
}
