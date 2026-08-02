use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2};
use super::instruction::{ExecContext, Instruction};

const TILE: usize = 16;

pub struct Matrix;

fn ceil_div(x: usize, d: usize) -> usize {
    (x + d - 1) / d
}

fn i8_at(bank: &[u8], row: usize, lane: usize) -> i8 {
    let row_bytes = bank_row_bytes();
    if lane >= row_bytes {
        panic!("matrix: i8 lane {lane} out of row_bytes={row_bytes}");
    }
    let off = row * row_bytes + lane;
    if off >= bank.len() {
        panic!("matrix: i8 read OOB row={row} lane={lane}");
    }
    bank[off] as i8
}

fn write_i32(bank: &mut [u8], row: usize, lane: usize, value: i32) {
    let row_bytes = bank_row_bytes();
    let off = row * row_bytes + lane * 4;
    if off + 4 > bank.len() {
        panic!("matrix: i32 write OOB row={row} lane={lane}");
    }
    bank[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

fn c_block(row: usize, n_tile: usize, m: usize, n: usize) -> usize {
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

impl Instruction for Matrix {
    const FUNCT: u32 = 65;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let op1_base = ((xs1 >> 30) & 0x7f) as usize;
        let op2_base = ((xs1 >> 37) & 0x7f) as usize;
        let wr_base = ((xs1 >> 44) & 0x7f) as usize;
        if xs1 >> 51 != 0 {
            panic!("matrix: rs1[63:51] must be 0");
        }

        let m = (xs2 & 0xfff) as usize;
        let n = ((xs2 >> 12) & 0xfff) as usize;
        let k = ((xs2 >> 24) & 0xfff) as usize;
        let mode = (xs2 >> 36) & 1;
        if xs2 >> 37 != 0 {
            panic!("matrix: rs2[63:37] must be 0");
        }
        let _ws = mode == 1;

        if op1 >= bank_num() as u64 || op2 >= bank_num() as u64 || wr >= bank_num() as u64 {
            panic!("matrix: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated
            || !ctx.cfgs[op2 as usize].allocated
            || !ctx.cfgs[wr as usize].allocated
        {
            panic!("matrix: bank not allocated");
        }
        if ctx.cfgs[op1 as usize].cols != 1 || ctx.cfgs[op2 as usize].cols != 1 {
            panic!(
                "matrix: op banks must have cols=1 (op1={} op2={})",
                ctx.cfgs[op1 as usize].cols,
                ctx.cfgs[op2 as usize].cols
            );
        }
        if ctx.cfgs[wr as usize].cols != 4 {
            panic!(
                "matrix: wr bank must be acc cols=4, got {}",
                ctx.cfgs[wr as usize].cols
            );
        }
        if m == 0 || n == 0 || k == 0 {
            panic!("matrix: M/N/K must be non-zero");
        }

        let k_tiles = ceil_div(k, TILE);
        let n_tiles = ceil_div(n, TILE);
        let a_rows = ceil_div(m, TILE) * k_tiles * TILE;
        let b_rows = n_tiles * k_tiles * TILE;
        let c_blocks = m * n_tiles;
        let lines = bank_lines();

        if op1_base + a_rows > lines {
            panic!("matrix: A range OOB base={op1_base} rows={a_rows} lines={lines}");
        }
        if op2_base + b_rows > lines {
            panic!("matrix: B range OOB base={op2_base} rows={b_rows} lines={lines}");
        }
        if wr_base + c_blocks > lines {
            panic!("matrix: C range OOB base={wr_base} blocks={c_blocks} lines={lines}");
        }

        let p1 = pbank(ctx.bank_map, op1);
        let p2 = pbank(ctx.bank_map, op2);
        let pw: Vec<_> = (0..4)
            .map(|g| pbank_group(ctx.bank_map, wr, g))
            .collect();

        let mut c = vec![vec![0i32; n]; m];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for kk in 0..k {
                    let mt = i / TILE;
                    let mr = i % TILE;
                    let kt = kk / TILE;
                    let kl = kk % TILE;
                    let a_row = op1_base + (mt * k_tiles + kt) * TILE + mr;
                    let a = i8_at(&ctx.banks[p1], a_row, kl);

                    let nt = j / TILE;
                    let nl = j % TILE;
                    let bkt = kk / TILE;
                    let bkr = kk % TILE;
                    let b_row = op2_base + (nt * k_tiles + bkt) * TILE + bkr;
                    let b = i8_at(&ctx.banks[p2], b_row, nl);

                    acc += a as i32 * b as i32;
                }
                c[i][j] = acc;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let nti = j / TILE;
                let lane = j % TILE;
                let block = wr_base + c_block(i, nti, m, n);
                let group = lane / 4;
                let sub = lane % 4;
                write_i32(&mut ctx.banks[pw[group]], block, sub, c[i][j]);
            }
        }
        0
    }

    fn latency(_xs1: u64, xs2: u64) -> u64 {
        let m = (xs2 & 0xfff).max(1);
        let n = ((xs2 >> 12) & 0xfff).max(1);
        let k = ((xs2 >> 24) & 0xfff).max(1);
        let mode = (xs2 >> 36) & 1;
        let body = m.saturating_mul(n).saturating_mul(k) / (TILE as u64);
        if mode == 1 {
            body + m.saturating_mul(n)
        } else {
            body + m.saturating_mul(n) / 2
        }
    }
}
