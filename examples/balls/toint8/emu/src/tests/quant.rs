use super::super::model::{quantize_f32, quantize_i32};

#[test]
fn rounds_ties_to_even_and_saturates() {
    assert_eq!(quantize_f32(2.5, 1.0, false), 2);
    assert_eq!(quantize_f32(3.5, 1.0, false), 4);
    assert_eq!(quantize_i32(1000, 1.0, false), 127);
    assert_eq!(quantize_i32(-1000, 1.0, false), -128);
    assert_eq!(quantize_i32(-7, 1.0, true), 0);
}
