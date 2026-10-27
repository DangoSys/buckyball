// F32AddBall contract: C = A + B over `rows` rows of four IEEE FP32 lanes,
// round to nearest even, every NaN result is the canonical 0x7fc00000.
// rs2[4:0] selects the source group of A; rs2[5] (first) replaces B with +0
// and B is not read.

pub const CANONICAL_NAN: u32 = 0x7fc0_0000;

pub fn bank_entries() -> u32 {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

pub fn validate(entries: u32, a: u32, b: u32, c: u32, rows: u32, group: u32, first: u32, cols: u32) {
    if a == b || a == c || b == c {
        panic!("f32add: banks must differ");
    }
    if group > 31 || first > 1 {
        panic!("f32add: rs2 fields out of range");
    }
    if group >= cols {
        panic!("f32add: source group must exist");
    }
    if rows == 0 || rows > entries {
        panic!("f32add: row span exceeds a bank");
    }
}

pub fn add_lane(source: u32, accumulator: u32) -> u32 {
    let sum = f32::from_bits(accumulator) + f32::from_bits(source);
    if sum.is_nan() {
        CANONICAL_NAN
    } else {
        sum.to_bits()
    }
}

pub fn add_word(source: u128, accumulator: u128, first: bool) -> u128 {
    let mut out = 0u128;
    for lane in 0..4 {
        let s = (source >> (32 * lane)) as u32;
        let acc = if first { 0 } else { (accumulator >> (32 * lane)) as u32 };
        out |= u128::from(add_lane(s, acc)) << (32 * lane);
    }
    out
}
