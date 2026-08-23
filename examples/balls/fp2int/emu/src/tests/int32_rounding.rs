use super::super::model::fp2int_i32_bits;

#[test]
fn rounds_to_nearest_even() {
    let scale = 0x3f80_0000;
    let cases = [
        (0x3f80_0000, 1),
        (0x4000_0000, 2),
        (0xbf80_0000, -1),
        (0x3f00_0000, 0),
        (0xbf00_0000, 0),
        (0x3fc0_0000, 2),
        (0xbfc0_0000, -2),
        (0x4020_0000, 2),
        (0xc020_0000, -2),
    ];

    for (input, expected) in cases {
        assert_eq!(fp2int_i32_bits(input, scale), expected);
    }
}
