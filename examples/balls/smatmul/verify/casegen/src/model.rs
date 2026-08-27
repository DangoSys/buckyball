pub const TILE: usize = 16;
pub const BANK_ROW_BYTES: usize = 16;
pub const VALUES_PER_WORD: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteExp {
    pub group: u32,
    pub addr: u32,
    pub data: [u8; 16],
    pub mask: u16,
}

pub fn pack_a(src: &[i8], rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(rows % TILE, 0);
    assert_eq!(k % TILE, 0);
    let mut packed = vec![0; rows * k];
    for row in 0..rows {
        for column in 0..k {
            let line = (row / TILE * (k / TILE) + column / TILE) * TILE + row % TILE;
            packed[line * TILE + column % TILE] = src[row * k + column] as u8;
        }
    }
    packed
}

pub fn pack_b_os(src: &[i8], k: usize) -> Vec<u8> {
    assert_eq!(k % TILE, 0);
    let mut packed = vec![0; k * TILE];
    for row in 0..k {
        for column in 0..TILE {
            packed[row * TILE + column] = src[row * TILE + column] as u8;
        }
    }
    packed
}

pub fn pack_b_ws(src: &[i8], k: usize, columns: usize) -> Vec<u8> {
    assert_eq!(k, TILE);
    assert_eq!(columns % TILE, 0);
    let mut packed = vec![0; k * columns];
    for panel in 0..columns / TILE {
        for row in 0..k {
            for column in 0..TILE {
                packed[(panel * TILE + row) * TILE + column] =
                    src[row * columns + panel * TILE + column] as u8;
            }
        }
    }
    packed
}

pub fn matmul(a: &[i8], b: &[i8], rows: usize, columns: usize, k: usize) -> Vec<Vec<i32>> {
    let mut c = vec![vec![0; columns]; rows];
    for row in 0..rows {
        for column in 0..columns {
            for reduction in 0..k {
                c[row][column] += a[row * k + reduction] as i32 * b[reduction * columns + column] as i32;
            }
        }
    }
    c
}

pub fn emit_writes(c: &[Vec<i32>], ws: bool, out_bw: usize) -> Vec<WriteExp> {
    assert!(matches!(out_bw, 1 | 2 | 4));
    let rows = c.len();
    let columns = c[0].len();
    assert_eq!(columns % TILE, 0);
    let rounds = VALUES_PER_WORD / out_bw;
    let mut writes = Vec::new();
    for panel in 0..columns / TILE {
        for row in 0..rows {
            for round in 0..rounds {
                let address = if ws {
                    panel * TILE * rounds + row * rounds + round
                } else {
                    (row / TILE * TILE + row % TILE) * rounds + round
                };
                for group in 0..out_bw {
                    let word = round * out_bw + group;
                    let mut data = [0; BANK_ROW_BYTES];
                    for lane in 0..VALUES_PER_WORD {
                        let value = c[row][panel * TILE + word * VALUES_PER_WORD + lane];
                        data[lane * 4..lane * 4 + 4].copy_from_slice(&value.to_le_bytes());
                    }
                    writes.push(WriteExp { group: group as u32, addr: address as u32, data, mask: 0xffff });
                }
            }
        }
    }
    writes
}

pub fn words(data: &[u8]) -> usize {
    assert_eq!(data.len() % BANK_ROW_BYTES, 0);
    data.len() / BANK_ROW_BYTES
}

pub fn encode_rs1(op1: u32, op2: u32, wr: u32) -> u64 {
    assert!(op1 < 1024 && op2 < 1024 && wr < 1024);
    u64::from(op1) | (u64::from(op2) << 10) | (u64::from(wr) << 20)
}

pub fn encode_rs2(rows: u32, columns: u32, k: u32) -> u64 {
    assert!(rows > 0 && columns > 0 && k > 0);
    assert!(rows <= 0xfff && columns <= 0xfff && k <= 0xfff);
    u64::from(rows) | (u64::from(columns) << 12) | (u64::from(k) << 24)
}
