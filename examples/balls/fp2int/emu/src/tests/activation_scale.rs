use super::super::model::{fp2int_da_from_max_abs_bits, fp32_divide};

#[test]
fn computes_activation_scale_with_fp32_division() {
    assert_eq!(
        fp32_divide(2.0f32.to_bits(), 127.0f32.to_bits()),
        0x3c81_0204
    );
    assert_eq!(fp2int_da_from_max_abs_bits(0), 1.0f32.to_bits());
    assert_eq!(fp2int_da_from_max_abs_bits(63.5f32.to_bits()), 0.5f32.to_bits());
}
