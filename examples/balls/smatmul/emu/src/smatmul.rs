use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

const TILE: usize = 16;

fn log2_up(n: usize) -> u32 {
    if n <= 1 {
        panic!("matrix: bank_lines must be > 1, got {n}");
    }
    usize::BITS - (n - 1).leading_zeros()
}

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

pub(crate) fn exec_smatmul(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let op2 = rs1_b1(xs1);
        let wr = rs1_b2(xs1);
        let rows = (xs2 & 0xfff) as usize;
        let cols = ((xs2 >> 12) & 0xfff) as usize;
        let k = ((xs2 >> 24) & 0xfff) as usize;
        if xs2 >> 36 != 0 {
            panic!("matrix: rs2[63:36] must be 0");
        }
        if rows == 0 || cols == 0 || k == 0 {
            panic!("matrix: rows/cols/k must be non-zero");
        }
        if cols > TILE {
            panic!("matrix: cols must be 1..{TILE}, got {cols}");
        }

        let lines = bank_lines();
        let addr_bits = log2_up(lines);
        if 3 * addr_bits > 34 {
            panic!("matrix: iter cannot hold 3 bases of {addr_bits} bits");
        }
        let iter = rs1_iter(xs1);
        let base_mask = (1u64 << addr_bits) - 1;
        let op1_base = (iter & base_mask) as usize;
        let op2_base = ((iter >> addr_bits) & base_mask) as usize;
        let wr_base = ((iter >> (2 * addr_bits)) & base_mask) as usize;
        if iter >> (3 * addr_bits) != 0 {
            panic!("matrix: iter unused high bits must be 0");
        }

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
                ctx.cfgs[op1 as usize].cols, ctx.cfgs[op2 as usize].cols
            );
        }
        if ctx.cfgs[wr as usize].cols != 4 {
            panic!(
                "matrix: wr bank must be acc cols=4, got {}",
                ctx.cfgs[wr as usize].cols
            );
        }

        let k_tiles = ceil_div(k, TILE);
        let a_rows = ceil_div(rows, TILE) * k_tiles * TILE;
        let b_rows = k_tiles * TILE;
        let c_blocks = rows;

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
        let pw: Vec<_> = (0..4).map(|g| pbank_group(ctx.bank_map, wr, g)).collect();

        let mut c = vec![vec![0i32; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0i32;
                for kk in 0..k {
                    let mt = i / TILE;
                    let mr = i % TILE;
                    let kt = kk / TILE;
                    let kl = kk % TILE;
                    let a_row = op1_base + (mt * k_tiles + kt) * TILE + mr;
                    let a = i8_at(&ctx.banks[p1], a_row, kl);

                    let bkr = kk % TILE;
                    let bkt = kk / TILE;
                    let b_row = op2_base + bkt * TILE + bkr;
                    let b = i8_at(&ctx.banks[p2], b_row, j);

                    acc += a as i32 * b as i32;
                }
                c[i][j] = acc;
            }
        }

        for i in 0..rows {
            for j in 0..cols {
                let block = wr_base + i;
                let group = j / 4;
                let sub = j % 4;
                write_i32(&mut ctx.banks[pw[group]], block, sub, c[i][j]);
            }
        }
        0
}

pub(crate) fn latency(_xs1: u64, xs2: u64) -> u64 {
        let rows = (xs2 & 0xfff) as u64;
        let cols = ((xs2 >> 12) & 0xfff) as u64;
        let k = ((xs2 >> 24) & 0xfff) as u64;
        if xs2 >> 36 != 0 {
            panic!("matrix: rs2[63:36] must be 0");
        }
        if rows == 0 || cols == 0 || k == 0 {
            panic!("matrix: rows/cols/k must be non-zero");
        }
        if cols > TILE as u64 {
            panic!("matrix: cols must be 1..{TILE}, got {cols}");
        }
        let body = rows.saturating_mul(cols).saturating_mul(k) / (TILE as u64);
        body + rows.saturating_mul(cols) / 2
}
