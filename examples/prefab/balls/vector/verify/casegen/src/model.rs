// VecBall VECMAT16 contract: C += A^T * B with int8 inputs and int32
// accumulators that wrap. op1 holds A^T and op2 holds B, `iter` rows of
// sixteen int8 each; the old C is read from the renamed bank wr_read_bank and
// the new C written to wr_bank, sixteen rows of sixteen int32 in four groups:
// group g holds columns 4g..4g+3 of each row. rs2 is reserved and zero.

pub const LANE: usize = 16;
pub const GROUPS: usize = 4;

pub fn bank_entries() -> u32 {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

pub fn validate(entries: u32, iter: u32) {
    if iter == 0 || iter > entries {
        panic!("vecmat16: iter must be 1..bank depth");
    }
}

pub fn i8_lane(word: u128, lane: usize) -> i32 {
    ((word >> (8 * lane)) as u8 as i8) as i32
}

// c[i][j] += sum_k at[k][i] * b[k][j], wrapping like the hardware.
pub fn vecmat16(at: &[u128], b: &[u128], c: &mut [[i32; LANE]; LANE]) {
    for i in 0..LANE {
        for j in 0..LANE {
            let mut acc = c[i][j];
            for k in 0..at.len() {
                acc = acc.wrapping_add(i8_lane(at[k], i).wrapping_mul(i8_lane(b[k], j)));
            }
            c[i][j] = acc;
        }
    }
}

// Word of group `g`, row `row`: columns 4g..4g+3, lowest column in the low bits.
pub fn c_word(c: &[[i32; LANE]; LANE], group: usize, row: usize) -> u128 {
    let mut word = 0u128;
    for lane in 0..4 {
        word |= u128::from(c[row][group * 4 + lane] as u32) << (32 * lane);
    }
    word
}
