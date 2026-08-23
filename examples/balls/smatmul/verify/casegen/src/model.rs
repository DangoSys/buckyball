pub const TILE: usize = 16;
pub const BANK_ROW_BYTES: usize = 16;
pub const WRITE_PORTS: usize = 4;
pub const ELEMS_PER_PORT: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteExp {
    pub group: u32,
    pub addr: u32,
    pub data: [u8; 16],
    pub mask: u16,
}

pub fn ceil_div(x: usize, d: usize) -> usize {
    (x + d - 1) / d
}

pub fn a_rows(m: usize, k: usize) -> usize {
    ceil_div(m, TILE) * ceil_div(k, TILE) * TILE
}

pub fn b_rows(n: usize, k: usize) -> usize {
    if n == 0 || n > TILE {
        panic!("b_rows: cols must be 1..{TILE}, got {n}");
    }
    ceil_div(k, TILE) * TILE
}

pub fn c_blocks(m: usize, n: usize) -> usize {
    if n == 0 || n > TILE {
        panic!("c_blocks: cols must be 1..{TILE}, got {n}");
    }
    m
}

pub fn pack_a(src: &[i8], m: usize, k: usize) -> Vec<u8> {
    let kt = ceil_div(k, TILE);
    let rows = a_rows(m, k);
    let mut dst = vec![0u8; rows * TILE];
    for r in 0..m {
        for c in 0..k {
            let mt = r / TILE;
            let mr = r % TILE;
            let kti = c / TILE;
            let lane = c % TILE;
            let bank_row = (mt * kt + kti) * TILE + mr;
            dst[bank_row * TILE + lane] = src[r * k + c] as u8;
        }
    }
    dst
}

pub fn pack_b(src: &[i8], k: usize, n: usize) -> Vec<u8> {
    let rows = b_rows(n, k);
    let mut dst = vec![0u8; rows * TILE];
    for r in 0..k {
        for c in 0..n {
            let lane = c;
            let kti = r / TILE;
            let kr = r % TILE;
            let bank_row = kti * TILE + kr;
            dst[bank_row * TILE + lane] = src[r * n + c] as u8;
        }
    }
    dst
}

pub fn matmul(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<Vec<i32>> {
    let mut c = vec![vec![0i32; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for kk in 0..k {
                acc += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            c[i][j] = acc;
        }
    }
    c
}

pub fn emit_writes(c: &[Vec<i32>], m: usize, n: usize) -> Vec<WriteExp> {
    if n == 0 || n > TILE {
        panic!("emit_writes: cols must be 1..{TILE}, got {n}");
    }
    let mut out = Vec::new();
    for row in 0..m {
        let block = row;
        let valid_elems = n;
        for port in 0..WRITE_PORTS {
            if port * ELEMS_PER_PORT >= valid_elems {
                continue;
            }
            let mut data = [0u8; 16];
            let mut mask = 0u16;
            for sub in 0..ELEMS_PER_PORT {
                let elem = port * ELEMS_PER_PORT + sub;
                let off = sub * 4;
                if elem < valid_elems {
                    let v = c[row][elem];
                    data[off..off + 4].copy_from_slice(&v.to_le_bytes());
                    for b in 0..4 {
                        mask |= 1u16 << (off + b);
                    }
                }
            }
            out.push(WriteExp {
                group: port as u32,
                addr: block as u32,
                data,
                mask,
            });
        }
    }
    out
}

pub fn num_words(total_bytes: usize) -> usize {
    assert_eq!(
        total_bytes % BANK_ROW_BYTES,
        0,
        "total_bytes not a multiple of bank row"
    );
    total_bytes / BANK_ROW_BYTES
}

pub fn words_from_rows(rows: usize) -> usize {
    num_words(rows * TILE)
}

pub fn encode_rs1(op1: u32, op2: u32, wr: u32) -> u64 {
    if op1 >= 1024 || op2 >= 1024 || wr >= 1024 {
        panic!("encode_rs1: bank out of 10-bit range");
    }
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20)
}

pub fn encode_rs2(rows: u32, cols: u32, k: u32) -> u64 {
    if rows == 0 || cols == 0 || k == 0 {
        panic!("encode_rs2: rows/cols/k must be non-zero");
    }
    if rows > 0xfff || k > 0xfff {
        panic!("encode_rs2: rows/k out of 12-bit range");
    }
    if cols > TILE as u32 {
        panic!("encode_rs2: cols must be 1..{TILE}, got {cols}");
    }
    u64::from(rows) | (u64::from(cols) << 12) | (u64::from(k) << 24)
}
