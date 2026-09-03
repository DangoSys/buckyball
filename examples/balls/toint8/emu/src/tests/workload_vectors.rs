use super::super::model::toint8_i8_bits;

#[test]
fn quantizes_workload_vectors() {
    let scale = 2.0f32.to_bits();
    let input = [
        0.125f32, -0.125, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 63.25, 63.75,
        -63.75, -64.75, 0.0, -0.0, 2.25, -2.25, 2.75, -2.75, 3.25, -3.25, 3.75, -3.75, 10.125,
        -10.125, 20.25, -20.25, 0.375, -0.375, 64.25, -65.25,
    ];
    let expected = [
        0i8, 0, 0, 0, 2, -2, 2, -2, 4, -4, 126, 127, -128, -128, 0, 0, 4, -4, 6, -6, 6, -6, 8,
        -8, 20, -20, 40, -40, 1, -1, 127, -128,
    ];

    for (value, expected) in input.into_iter().zip(expected) {
        assert_eq!(toint8_i8_bits(value.to_bits(), scale), expected, "input={value}");
    }
}
