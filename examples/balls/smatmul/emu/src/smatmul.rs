use super::super::bank::{bank_lines, bank_num, bank_row_bytes};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use std::sync::{Mutex, OnceLock};

const TILE: usize = 16;

#[derive(Default)]
struct State {
    bias: Option<[i32; TILE]>,
    chain: Option<Chain>,
}

struct Chain {
    rows: usize,
    cols: usize,
    output_bank: u64,
    accumulators: Vec<i32>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

fn smatmul_state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State::default()))
}

fn read_i32(bank: &[u8], row: usize, lane: usize) -> i32 {
    let offset = row * bank_row_bytes() + lane * 4;
    i32::from_le_bytes(bank[offset..offset + 4].try_into().unwrap())
}

fn write_i32(bank: &mut [u8], row: usize, lane: usize, value: i32) {
    let offset = row * bank_row_bytes() + lane * 4;
    bank[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(crate) fn exec_bias(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let bank = rs1_b0(xs1);
    if rs1_b1(xs1) != 0 || rs1_b2(xs1) != 0 || rs1_iter(xs1) != 4 || xs2 != 0 {
        panic!("smatmul_bias: bank1/bank2/rs2 must be zero and iter must be four");
    }
    if bank >= bank_num() as u64 {
        panic!("smatmul_bias: invalid bank id");
    }
    if !ctx.cfgs[bank as usize].allocated || ctx.cfgs[bank as usize].cols != 1 {
        panic!("smatmul_bias: bias bank must be one allocated column");
    }
    let mut state = smatmul_state().lock().unwrap();
    if state.chain.is_some() {
        panic!("smatmul_bias: cannot replace bias during an accumulation chain");
    }
    let physical = pbank(ctx.bank_map, bank);
    let mut bias = [0i32; TILE];
    for group in 0..4 {
        for lane in 0..4 {
            bias[group * 4 + lane] = read_i32(&ctx.banks[physical], group, lane);
        }
    }
    state.bias = Some(bias);
    0
}

pub(crate) fn exec_smatmul(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let a_bank = rs1_b0(xs1);
    let b_bank = rs1_b1(xs1);
    let c_bank = rs1_b2(xs1);
    let rows = (xs2 & 0xfff) as usize;
    let cols = ((xs2 >> 12) & 0xfff) as usize;
    let k = rs1_iter(xs1) as usize;
    let first = (xs2 >> 24) & 1 != 0;
    let last = (xs2 >> 25) & 1 != 0;
    if xs2 >> 26 != 0 {
        panic!("smatmul: rs2[63:26] must be zero");
    }
    if (rows != 1 && (rows == 0 || rows % TILE != 0))
        || cols != TILE
        || k == 0
        || k % TILE != 0
    {
        panic!("smatmul: M must be one or a positive multiple of 16, K must be a positive multiple of 16, and N must be 16");
    }
    if a_bank >= bank_num() as u64 || b_bank >= bank_num() as u64 || c_bank >= bank_num() as u64 {
        panic!("smatmul: invalid bank id");
    }
    if a_bank == b_bank || a_bank == c_bank || b_bank == c_bank {
        panic!("smatmul: A, B, and C banks must differ");
    }
    if !ctx.cfgs[a_bank as usize].allocated
        || !ctx.cfgs[b_bank as usize].allocated
        || !ctx.cfgs[c_bank as usize].allocated
    {
        panic!("smatmul: bank not allocated");
    }
    if ctx.cfgs[a_bank as usize].cols != 1
        || ctx.cfgs[b_bank as usize].cols != 1
        || ctx.cfgs[c_bank as usize].cols != 1
    {
        panic!("smatmul: A, B, and C banks must each have one column");
    }

    let a_rows = if rows == 1 { k / TILE } else { rows * k / TILE };
    let b_rows = k;
    let c_rows = rows * 4;
    if a_rows > bank_lines() || b_rows > bank_lines() || c_rows > bank_lines() {
        panic!("smatmul: bank footprint exceeds bank depth");
    }

    let mut state = smatmul_state().lock().unwrap();
    let mut chain = if first {
        if state.chain.is_some() {
            panic!("smatmul: first block issued while another chain is live");
        }
        let bias = state
            .bias
            .unwrap_or_else(|| panic!("smatmul: bias must be preloaded before first block"));
        let mut accumulators = vec![0i32; rows * cols];
        for row in 0..rows {
            accumulators[row * cols..(row + 1) * cols].copy_from_slice(&bias);
        }
        Chain {
            rows,
            cols,
            output_bank: c_bank,
            accumulators,
        }
    } else {
        state
            .chain
            .take()
            .unwrap_or_else(|| panic!("smatmul: continuation has no live chain"))
    };
    if chain.rows != rows
        || chain.cols != cols
        || chain.output_bank != c_bank
    {
        panic!("smatmul: continuation changed output shape or destination");
    }

    let pa = pbank(ctx.bank_map, a_bank);
    let pb = pbank(ctx.bank_map, b_bank);
    let k_tiles = k / TILE;
    for row in 0..rows {
        for col in 0..cols {
            let acc = &mut chain.accumulators[row * cols + col];
            for inner in 0..k {
                let a_row = if rows == 1 {
                    inner / TILE
                } else {
                    ((row / TILE) * k_tiles + inner / TILE) * TILE + row % TILE
                };
                let b_row = (inner / TILE) * TILE + inner % TILE;
                let a = ctx.banks[pa][a_row * bank_row_bytes() + inner % TILE] as i8;
                let b = ctx.banks[pb][b_row * bank_row_bytes() + col] as i8;
                *acc = acc.wrapping_add((a as i32).wrapping_mul(b as i32));
            }
        }
    }

    if last {
        let pc = pbank(ctx.bank_map, c_bank);
        for row in 0..rows {
            for col in 0..cols {
                write_i32(
                    &mut ctx.banks[pc],
                    row * 4 + col / 4,
                    col % 4,
                    chain.accumulators[row * cols + col],
                );
            }
        }
    } else {
        state.chain = Some(chain);
    }
    0
}

pub(crate) fn bias_latency(xs1: u64, xs2: u64) -> u64 {
    if rs1_b1(xs1) != 0 || rs1_b2(xs1) != 0 || rs1_iter(xs1) != 4 || xs2 != 0 {
        panic!("smatmul_bias: unused fields must be zero");
    }
    4
}

pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let rows = xs2 & 0xfff;
    let cols = (xs2 >> 12) & 0xfff;
    let k = rs1_iter(xs1);
    if xs2 >> 26 != 0
        || (rows != 1 && (rows == 0 || rows % 16 != 0))
        || cols != 16
        || k == 0
        || k % 16 != 0
    {
        panic!("smatmul: illegal matrix encoding");
    }
    rows * cols * k / 16 + if (xs2 >> 25) & 1 != 0 { rows * 4 } else { 0 }
}
