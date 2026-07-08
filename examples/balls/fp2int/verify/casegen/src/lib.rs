mod casegen;
#[path = "../../../emu/src/model.rs"]
mod model;

#[no_mangle]
pub extern "C" fn fp2int_ref_i32(fp_bits: u32, scale_bits: u32) -> i32 {
  model::fp2int_i32_bits(fp_bits, scale_bits)
}

#[no_mangle]
pub extern "C" fn fp2int_ref_i8(fp_bits: u32, scale_bits: u32) -> i32 {
  i32::from(model::fp2int_i8_bits(fp_bits, scale_bits))
}

#[no_mangle]
pub extern "C" fn fp2int_case_bid(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).bid
}

#[no_mangle]
pub extern "C" fn fp2int_case_funct7(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).funct7
}

#[no_mangle]
pub extern "C" fn fp2int_case_iter(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).iter
}

#[no_mangle]
pub extern "C" fn fp2int_case_scale_bits(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).scale_bits
}

#[no_mangle]
pub extern "C" fn fp2int_case_op1_bank(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).op1_bank
}

#[no_mangle]
pub extern "C" fn fp2int_case_op2_bank(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).op2_bank
}

#[no_mangle]
pub extern "C" fn fp2int_case_wr_bank(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).wr_bank
}

#[no_mangle]
pub extern "C" fn fp2int_case_op1_col(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).op1_col
}

#[no_mangle]
pub extern "C" fn fp2int_case_op2_col(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).op2_col
}

#[no_mangle]
pub extern "C" fn fp2int_case_wr_col(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).wr_col
}

#[no_mangle]
pub extern "C" fn fp2int_case_meta_bank(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).meta_bank
}

#[no_mangle]
pub extern "C" fn fp2int_case_rob_id(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).rob_id
}

#[no_mangle]
pub extern "C" fn fp2int_case_is_sub(seed: u32, index: u32) -> u32 {
  u32::from(casegen::gen_case(seed, index).is_sub)
}

#[no_mangle]
pub extern "C" fn fp2int_case_sub_rob_id(seed: u32, index: u32) -> u32 {
  casegen::gen_case(seed, index).sub_rob_id
}

#[no_mangle]
pub extern "C" fn fp2int_case_word_lo(seed: u32, index: u32, word_index: u32) -> u64 {
  casegen::gen_case(seed, index).word_lo(word_index as usize)
}

#[no_mangle]
pub extern "C" fn fp2int_case_word_hi(seed: u32, index: u32, word_index: u32) -> u64 {
  casegen::gen_case(seed, index).word_hi(word_index as usize)
}
