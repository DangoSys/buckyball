pub const TILE: usize = 16;
pub const ROW_BYTES: usize = 16;
pub const RESULT_WORDS: usize = 4;
pub const BANK_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteExp {
    pub addr: u32,
    pub data: [u8; ROW_BYTES],
}

pub fn pack_a(src: &[i8], rows: usize) -> Vec<u8> {
    assert_eq!(src.len(), rows * TILE);
    src.iter().map(|value| *value as u8).collect()
}

pub fn pack_b(src: &[i8]) -> Vec<u8> {
    assert_eq!(src.len(), TILE * TILE);
    src.iter().map(|value| *value as u8).collect()
}

pub fn emit_writes(c: &[Vec<i32>]) -> Vec<WriteExp> {
    let mut writes = Vec::with_capacity(c.len() * RESULT_WORDS);
    for (row, values) in c.iter().enumerate() {
        assert_eq!(values.len(), TILE);
        for word in 0..RESULT_WORDS {
            let mut data = [0; ROW_BYTES];
            for lane in 0..4 {
                let value = values[word * 4 + lane];
                data[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            writes.push(WriteExp {
                addr: (row * RESULT_WORDS + word) as u32,
                data,
            });
        }
    }
    writes
}

pub fn word(data: &[u8], index: usize) -> [u8; ROW_BYTES] {
    data[index * ROW_BYTES..(index + 1) * ROW_BYTES]
        .try_into()
        .unwrap()
}

pub fn encode_rs1(op1: u32, op2: u32, wr: u32, iter: u64) -> u64 {
    assert!(op1 < 1024 && op2 < 1024 && wr < 1024 && iter < (1_u64 << 34));
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20) | (iter << 30)
}

pub fn encode_rs2(rows: usize, first: bool, last: bool) -> u64 {
    assert!(matches!(rows, 16 | 32 | 64));
    rows as u64
        | ((TILE as u64) << 12)
        | ((first as u64) << 24)
        | ((last as u64) << 25)
}
