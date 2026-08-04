#[path = "../../../emu/src/model.rs"]
mod model;

#[no_mangle]
pub extern "C" fn int2fp_ref_fp32(value: i32, scale_bits: u32) -> u32 {
  model::int2fp_fp32_bits(value, scale_bits)
}
