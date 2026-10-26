#[no_mangle]
pub extern "C" fn decoder_ref_domain(
    funct: u32,
    opcode: u32,
    rs1: u64,
    rvv: u32,
    banks: u32,
    bits: u32,
) -> u32 {
    let bank = rs1 & ((1u64 << bits) - 1);
    assert!(funct != 0 && ![0x57, 7, 0x27].contains(&opcode));
    if funct == 1 {
        0
    } else if [44, 79].contains(&funct)
        || (rvv != 0 && funct == 32 && bank >= banks as u64 && bank < (banks + 2) as u64)
    {
        2
    } else if [13, 16, 32, 33, 34, 35].contains(&funct) {
        1
    } else {
        3
    }
}
