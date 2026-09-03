pub fn quantize_f32(value: f32, scale: f32, relu: bool) -> i8 {
    assert!(value.is_finite(), "quant: input must be finite");
    assert!(scale.is_finite() && scale > 0.0, "quant: scale must be finite and positive");
    let scaled = value * scale;
    assert!(scaled.is_finite(), "quant: scaled value must be finite");
    let rounded = scaled.round_ties_even();
    let clamped = if relu { rounded.max(0.0) } else { rounded }.clamp(-128.0, 127.0);
    clamped as i8
}

pub fn quantize_i32(value: i32, scale: f32, relu: bool) -> i8 {
    quantize_f32(value as f32, scale, relu)
}
