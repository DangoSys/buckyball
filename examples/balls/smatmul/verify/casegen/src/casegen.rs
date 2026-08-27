use crate::model;

pub const MAX_WORDS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixCmd {
    pub bid: u32,
    pub ws: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub rob_id: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub num_a_words: u32,
    pub num_b_words: u32,
    pub num_writes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixCase {
    pub cmd: MatrixCmd,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub writes: Vec<model::WriteExp>,
}

impl MatrixCase {
    pub fn a_word_lo(&self, index: usize) -> u64 { word_lo(&self.a, index) }
    pub fn a_word_hi(&self, index: usize) -> u64 { word_hi(&self.a, index) }
    pub fn b_word_lo(&self, index: usize) -> u64 { word_lo(&self.b, index) }
    pub fn b_word_hi(&self, index: usize) -> u64 { word_hi(&self.b, index) }
}

fn word_lo(data: &[u8], index: usize) -> u64 {
    let offset = index * model::BANK_ROW_BYTES;
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn word_hi(data: &[u8], index: usize) -> u64 {
    let offset = index * model::BANK_ROW_BYTES + 8;
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

pub fn gen_case(seed: u32, index: u32, bid: u32, out_bw: usize) -> MatrixCase {
    match index {
        0 => build_case(bid, false, 16, 16, 16, out_bw, seed, 7),
        1 => build_case(bid, false, 32, 16, 16, out_bw, seed, 8),
        2 => build_case(bid, true, 16, 16, 16, out_bw, seed, 9),
        3 => build_case(bid, true, 16, 32, 16, out_bw, seed, 10),
        _ => random_case(seed, index, bid, out_bw),
    }
}

fn build_case(bid: u32, ws: bool, rows: usize, columns: usize, k: usize,
              out_bw: usize, seed: u32, rob_id: u32) -> MatrixCase {
    assert!(matches!(out_bw, 1 | 2 | 4));
    assert_eq!(rows % model::TILE, 0);
    assert_eq!(columns % model::TILE, 0);
    assert_eq!(k % model::TILE, 0);
    if ws { assert_eq!(rows, model::TILE); assert_eq!(k, model::TILE); }
    let mut rng = Rng::new(seed, rows as u32 ^ columns as u32 ^ k as u32);
    let a: Vec<i8> = (0..rows * k).map(|_| rng.value()).collect();
    let b: Vec<i8> = (0..k * columns).map(|_| rng.value()).collect();
    let a_packed = model::pack_a(&a, rows, k);
    let b_packed = if ws { model::pack_b_ws(&b, k, columns) } else { model::pack_b_os(&b, k) };
    let writes = model::emit_writes(&model::matmul(&a, &b, rows, columns, k), ws, out_bw);
    assert!(model::words(&a_packed) <= MAX_WORDS);
    assert!(model::words(&b_packed) <= MAX_WORDS);
    MatrixCase {
        cmd: MatrixCmd {
            bid, ws: ws as u32, m: rows as u32, n: columns as u32, k: k as u32,
            op1_bank: 0, op2_bank: 1, wr_bank: 2, rob_id,
            rs1_lo: model::encode_rs1(0, 1, 2) as u32,
            rs1_hi: (model::encode_rs1(0, 1, 2) >> 32) as u32,
            rs2_lo: model::encode_rs2(rows as u32, columns as u32, k as u32) as u32,
            rs2_hi: (model::encode_rs2(rows as u32, columns as u32, k as u32) >> 32) as u32,
            num_a_words: model::words(&a_packed) as u32,
            num_b_words: model::words(&b_packed) as u32,
            num_writes: writes.len() as u32,
        },
        a: a_packed, b: b_packed, writes,
    }
}

fn random_case(seed: u32, index: u32, bid: u32, out_bw: usize) -> MatrixCase {
    let mut rng = Rng::new(seed, index);
    let ws = rng.next() & 1 != 0;
    let rows = if ws { 16 } else if rng.next() & 1 == 0 { 16 } else { 32 };
    let columns = if ws && rng.next() & 1 != 0 { 32 } else { 16 };
    build_case(bid, ws, rows, columns, 16, out_bw, rng.next(), rng.next() & 15)
}

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u32, salt: u32) -> Self { Self { state: u64::from(seed) << 32 | u64::from(salt) } }
    fn next(&mut self) -> u32 { self.state ^= self.state << 13; self.state ^= self.state >> 7; self.state ^= self.state << 17; self.state as u32 }
    fn value(&mut self) -> i8 { (self.next() % 23) as i8 - 11 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn os_uses_two_compact_rounds() {
        let case = gen_case(1, 0, 1, 2);
        assert_eq!(case.cmd.ws, 0);
        assert_eq!(case.cmd.num_writes, 32);
        assert_eq!(case.writes[0].group, 0);
        assert_eq!(case.writes[1].group, 1);
        assert_eq!(case.writes[2].addr, 1);
    }
    #[test]
    fn ws_second_panel_follows_first() {
        let case = gen_case(1, 3, 1, 2);
        assert_eq!(case.cmd.ws, 1);
        assert_eq!(case.cmd.num_b_words, 32);
        assert_eq!(case.writes[32].addr, 32);
    }
}
