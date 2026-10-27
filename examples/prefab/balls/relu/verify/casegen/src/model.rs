// ReluBall contract: max(x, 0) on four i32 lanes per row, in place on one
// bank group. The Ball walks segments 0, stride, 2 * stride, ... up to the
// physical bank depth and updates the first `iter` rows of each segment.

pub fn bank_entries() -> u32 {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

pub fn validate(entries: u32, iter: u32, stride: u64, group: u32, cols: u32) {
    if iter == 0 || iter % 16 != 0 {
        panic!("relu: iter must be a positive multiple of 16");
    }
    if stride == 0 || stride > u64::from(entries) || u64::from(entries) % stride != 0 {
        panic!("relu: stride must divide the physical bank depth");
    }
    if u64::from(iter) > stride {
        panic!("relu: iter must not exceed stride");
    }
    if cols == 0 || group >= cols {
        panic!("relu: group is not allocated in bank");
    }
}

// Rows in the order the Ball reads and rewrites them.
pub fn rows(entries: u32, iter: u32, stride: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for segment in (0..entries).step_by(stride as usize) {
        for line in 0..iter {
            out.push(segment + line);
        }
    }
    out
}

pub fn relu_word(word: u128) -> u128 {
    let mut out = 0u128;
    for lane in 0..4 {
        let value = (word >> (32 * lane)) as u32 as i32;
        out |= u128::from(value.max(0) as u32) << (32 * lane);
    }
    out
}
