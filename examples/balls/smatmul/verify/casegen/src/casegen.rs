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
    pub fn a_word_lo(&self, word_index: usize) -> u64 {
        word_lo(&self.a, word_index)
    }
    pub fn a_word_hi(&self, word_index: usize) -> u64 {
        word_hi(&self.a, word_index)
    }
    pub fn b_word_lo(&self, word_index: usize) -> u64 {
        word_lo(&self.b, word_index)
    }
    pub fn b_word_hi(&self, word_index: usize) -> u64 {
        word_hi(&self.b, word_index)
    }
}

fn word_lo(data: &[u8], word_index: usize) -> u64 {
    let off = word_index * model::BANK_ROW_BYTES;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

fn word_hi(data: &[u8], word_index: usize) -> u64 {
    let off = word_index * model::BANK_ROW_BYTES + 8;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(buf)
}

pub fn gen_case(seed: u32, index: u32, bid: u32) -> MatrixCase {
    match index {
        0 => directed_4x4(bid),
        1 => directed_5x7x3(bid),
        2 => directed_16x16(bid),
        3 => directed_32x16(bid),
        _ => random_case(seed, index, bid),
    }
}

fn build_case(
    bid: u32,
    m: usize,
    n: usize,
    k: usize,
    op1_bank: u32,
    op2_bank: u32,
    wr_bank: u32,
    rob_id: u32,
    a_flat: &[i8],
    b_flat: &[i8],
) -> MatrixCase {
    if op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank {
        panic!("build_case: banks must be distinct");
    }
    if op1_bank > 7 || op2_bank > 7 || wr_bank > 7 {
        panic!("build_case: banks must be in 0..7");
    }
    if a_flat.len() != m * k || b_flat.len() != k * n {
        panic!("build_case: flat buffer size mismatch");
    }

    let a = model::pack_a(a_flat, m, k);
    let b = model::pack_b(b_flat, k, n);
    let c = model::matmul(a_flat, b_flat, m, n, k);
    let writes = model::emit_writes(&c, m, n);

    let na = model::words_from_rows(model::a_rows(m, k)) as u32;
    let nb = model::words_from_rows(model::b_rows(n, k)) as u32;
    if na as usize > MAX_WORDS || nb as usize > MAX_WORDS {
        panic!("build_case: word count out of range a={na} b={nb}");
    }

    let rs1 = model::encode_rs1(op1_bank, op2_bank, wr_bank);
    let rs2 = model::encode_rs2(m as u32, n as u32, k as u32);

    let ws = ((m + 15) / 16 >= 2) as u32;

    MatrixCase {
        cmd: MatrixCmd {
            bid,
            ws,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            op1_bank,
            op2_bank,
            wr_bank,
            rob_id,
            rs1_lo: (rs1 & 0xffff_ffff) as u32,
            rs1_hi: (rs1 >> 32) as u32,
            rs2_lo: (rs2 & 0xffff_ffff) as u32,
            rs2_hi: (rs2 >> 32) as u32,
            num_a_words: na,
            num_b_words: nb,
            num_writes: writes.len() as u32,
        },
        a,
        b,
        writes,
    }
}

fn directed_4x4(bid: u32) -> MatrixCase {
    let m = 4usize;
    let n = 4usize;
    let k = 4usize;
    let a: Vec<i8> = (1..=16).map(|x| x as i8).collect();
    let mut b = vec![0i8; k * n];
    for i in 0..k {
        b[i * n + i] = 1;
    }
    build_case(bid, m, n, k, 0, 1, 2, 3, &a, &b)
}

fn directed_5x7x3(bid: u32) -> MatrixCase {
    let m = 5usize;
    let n = 7usize;
    let k = 3usize;
    let a: Vec<i8> = (1..=15).map(|x| x as i8).collect();
    let mut b = vec![0i8; k * n];
    for i in 0..k {
        b[i * n + i] = 1;
    }
    build_case(bid, m, n, k, 0, 1, 2, 3, &a, &b)
}

fn directed_16x16(bid: u32) -> MatrixCase {
    let m = 16usize;
    let n = 16usize;
    let k = 16usize;
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 127) + 1) as i8).collect();
    let mut b = vec![0i8; k * n];
    for i in 0..k {
        for j in 0..n {
            b[i * n + j] = if i == j { 1 } else { 0 };
        }
    }
    build_case(bid, m, n, k, 0, 1, 2, 7, &a, &b)
}

fn directed_32x16(bid: u32) -> MatrixCase {
    let m = 32usize;
    let n = 16usize;
    let k = 16usize;
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 127) + 1) as i8).collect();
    let mut b = vec![0i8; k * n];
    for i in 0..k {
        for j in 0..n {
            b[i * n + j] = if i == j { 1 } else { 0 };
        }
    }
    build_case(bid, m, n, k, 0, 1, 2, 7, &a, &b)
}

fn random_case(seed: u32, index: u32, bid: u32) -> MatrixCase {
    let mut rng = Rng::new(seed, index);
    let m_pool = [1usize, 2, 4, 8, 16, 32];
    let nk_pool = [1usize, 2, 4, 8, 16];
    let m = m_pool[(rng.next() as usize) % m_pool.len()];
    let n = nk_pool[(rng.next() as usize) % nk_pool.len()];
    let k = nk_pool[(rng.next() as usize) % nk_pool.len()];

    let op1_bank = rng.range(0, 7);
    let mut op2_bank = rng.range(0, 7);
    while op2_bank == op1_bank {
        op2_bank = rng.range(0, 7);
    }
    let mut wr_bank = rng.range(0, 7);
    while wr_bank == op1_bank || wr_bank == op2_bank {
        wr_bank = rng.range(0, 7);
    }

    let a: Vec<i8> = (0..m * k)
        .map(|_| (rng.next() & 0x7f) as i8)
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|_| (rng.next() & 0x7f) as i8)
        .collect();

    build_case(
        bid,
        m,
        n,
        k,
        op1_bank,
        op2_bank,
        wr_bank,
        rng.range(0, 15),
        &a,
        &b,
    )
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u32, index: u32) -> Self {
        let state = (u64::from(seed) << 32) ^ u64::from(index) ^ 0x9E37_79B9_7F4A_7C15;
        Self { state }
    }
    fn next(&mut self) -> u32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 32) as u32
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model;

    #[test]
    fn pack_a_matches_ctest_layout() {
        let m = 4;
        let k = 4;
        let a: Vec<i8> = (1..=16).map(|x| x as i8).collect();
        let pa = model::pack_a(&a, m, k);
        assert_eq!(pa.len(), model::a_rows(m, k) * model::TILE);
        assert_eq!(pa[0], 1);
        assert_eq!(pa[1], 2);
        assert_eq!(pa[2], 3);
        assert_eq!(pa[3], 4);
    }

    #[test]
    fn pack_b_identity() {
        let k = 4;
        let n = 4;
        let mut b = vec![0i8; k * n];
        for i in 0..k {
            b[i * n + i] = 1;
        }
        let pb = model::pack_b(&b, k, n);
        assert_eq!(pb[0], 1);
        assert_eq!(pb[17], 1);
    }

    #[test]
    fn matmul_4x4_identity_b() {
        let a: Vec<i8> = (1..=16).map(|x| x as i8).collect();
        let mut b = vec![0i8; 16];
        for i in 0..4 {
            b[i * 4 + i] = 1;
        }
        let c = model::matmul(&a, &b, 4, 4, 4);
        for r in 0..4 {
            for col in 0..4 {
                assert_eq!(c[r][col], a[r * 4 + col] as i32);
            }
        }
    }

    #[test]
    fn directed_os_4x4_shape() {
        let case = gen_case(0, 0, 1);
        assert_eq!(case.cmd.ws, 0);
        assert_eq!(case.cmd.m, 4);
        assert_eq!(case.cmd.n, 4);
        assert_eq!(case.cmd.k, 4);
        assert_eq!(case.cmd.op1_bank, 0);
        assert_eq!(case.cmd.op2_bank, 1);
        assert_eq!(case.cmd.wr_bank, 2);
        assert_eq!(case.cmd.num_a_words, 16);
        assert_eq!(case.cmd.num_b_words, 16);
        assert_eq!(case.cmd.num_writes, 4);
        assert_eq!(case.writes.len(), 4);
        for w in &case.writes {
            assert_eq!(w.group, 0);
            assert_eq!(w.mask, 0xFFFF);
        }
    }

    #[test]
    fn directed_5x7x3_shape() {
        let case = gen_case(0, 1, 1);
        assert_eq!(case.cmd.m, 5);
        assert_eq!(case.cmd.n, 7);
        assert_eq!(case.cmd.k, 3);
        assert_eq!(case.cmd.num_writes, 10);
    }

    #[test]
    fn directed_16x16_os_writes() {
        let case = gen_case(0, 2, 1);
        assert_eq!(case.cmd.m, 16);
        assert_eq!(case.cmd.ws, 0);
        assert_eq!(case.cmd.num_writes, 64);
        assert!(case.cmd.num_a_words < 128);
        assert!(case.cmd.num_b_words < 128);
    }

    #[test]
    fn directed_32x16_is_ws() {
        let case = gen_case(0, 3, 1);
        assert_eq!(case.cmd.m, 32);
        assert_eq!(case.cmd.n, 16);
        assert_eq!(case.cmd.k, 16);
        assert_eq!(case.cmd.ws, 1);
        assert_eq!(case.cmd.num_a_words, 32);
    }

    #[test]
    fn random_deterministic() {
        let a = gen_case(0xCAFE_BABE, 5, 4);
        let b = gen_case(0xCAFE_BABE, 5, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn bid_required_arg() {
        let c0 = gen_case(0, 0, 1);
        let c1 = gen_case(0, 0, 4);
        assert_eq!(c0.cmd.bid, 1);
        assert_eq!(c1.cmd.bid, 4);
    }

    #[test]
    fn rs1_rs2_encoding() {
        let case = gen_case(0, 0, 1);
        let rs1 = (u64::from(case.cmd.rs1_hi) << 32) | u64::from(case.cmd.rs1_lo);
        let rs2 = (u64::from(case.cmd.rs2_hi) << 32) | u64::from(case.cmd.rs2_lo);
        assert_eq!(rs1, model::encode_rs1(0, 1, 2));
        assert_eq!(rs2, model::encode_rs2(4, 4, 4));
    }

    #[test]
    fn write_mask_tail_n5() {
        let a = vec![1i8; 5];
        let b = vec![1i8; 25];
        let c = model::matmul(&a, &b, 1, 5, 5);
        let writes = model::emit_writes(&c, 1, 5);
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].mask, 0xFFFF);
        assert_eq!(writes[1].mask, 0x000F);
    }
}
