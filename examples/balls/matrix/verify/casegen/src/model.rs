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
    ceil_div(n, TILE) * ceil_div(k, TILE) * TILE
}

pub fn c_blocks(m: usize, n: usize) -> usize {
    m * ceil_div(n, TILE)
}

pub fn c_block(row: usize, n_tile: usize, m: usize, n: usize) -> usize {
    let n_tiles = ceil_div(n, TILE);
    let mt = row / TILE;
    let mr = row % TILE;
    let mut block = 0usize;
    for t in 0..mt {
        let rows = (m - t * TILE).min(TILE);
        block += rows * n_tiles;
    }
    let rows = (m - mt * TILE).min(TILE);
    block + n_tile * rows + mr
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
    let kt = ceil_div(k, TILE);
    let rows = b_rows(n, k);
    let mut dst = vec![0u8; rows * TILE];
    for r in 0..k {
        for c in 0..n {
            let nt = c / TILE;
            let lane = c % TILE;
            let kti = r / TILE;
            let kr = r % TILE;
            let bank_row = (nt * kt + kti) * TILE + kr;
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
    let n_tiles = ceil_div(n, TILE);
    let mut out = Vec::new();
    for row in 0..m {
        for nti in 0..n_tiles {
            let block = c_block(row, nti, m, n);
            let valid_elems = (n - nti * TILE).min(TILE);
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
                        let col = nti * TILE + elem;
                        let v = c[row][col];
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

pub fn encode_rs2(m: u32, n: u32, k: u32, mode: u32) -> u64 {
    if m == 0 || n == 0 || k == 0 {
        panic!("encode_rs2: M/N/K must be non-zero");
    }
    if m > 0xfff || n > 0xfff || k > 0xfff {
        panic!("encode_rs2: M/N/K out of 12-bit range");
    }
    if mode > 1 {
        panic!("encode_rs2: mode must be 0 or 1");
    }
    u64::from(m) | (u64::from(n) << 12) | (u64::from(k) << 24) | (u64::from(mode) << 36)
}
