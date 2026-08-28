use super::super::model::fp2int_i8_bits;

#[test]
fn saturates_beyond_int8_range() {
    let scale = 0x3f80_0000;

    assert_eq!(fp2int_i8_bits(0x4300_0000, scale), 127);
    assert_eq!(fp2int_i8_bits(0xc300_0000, scale), -128);
}
