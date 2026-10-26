use bebop_bemu::thread::banks::{bank_lines, bank_row_bytes};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use std::collections::HashMap;

fn tile_rows(instance: u32) -> usize { bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::smatmul::Params>("smatmul", instance).tile_rows.expect("tileRows") as usize }
fn tile_cols(instance: u32) -> usize { bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::smatmul::Params>("smatmul", instance).tile_cols.expect("tileCols") as usize }

#[derive(Default)]
struct State {
    bias: Option<Vec<i32>>,
    chain: Option<Chain>,
}

struct Chain {
    rows: usize,
    cols: usize,
    output_bank: u64,
    output_base: usize,
    accumulators: Vec<i32>,
}



fn read_i32(bank: &[u8], row: usize, lane: usize) -> i32 {
    let offset = row * bank_row_bytes() + lane * 4;
    i32::from_le_bytes(bank[offset..offset + 4].try_into().unwrap())
}

pub(crate) fn exec_bias(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let ball_state = ctx.state::<HashMap<usize, State>>();
    let bank = rs1_b0(xs1);
    let input_base = (xs2 & 0x3f) as usize;
    if rs1_b1(xs1) != 0 || rs1_b2(xs1) != 0 || rs1_iter(xs1) != 4 || xs2 >> 6 != 0 {
        panic!("smatmul_bias: bank1/bank2/rs2[63:6] must be zero and iter must be four");
    }
    if !ctx.config(bank).allocated || ctx.config(bank).cols != 1 {
        panic!("smatmul_bias: bias bank must be one allocated column");
    }
    if input_base + 4 > bank_lines() {
        panic!("smatmul_bias: inputBase plus four rows exceeds bank depth");
    }
    let physical = pbank(ctx, bank);
    let mut bias = vec![0i32; tile_cols(ctx.ball_id.expect("Ball instance"))];
    for group in 0..4 {
        for lane in 0..4 {
            bias[group * 4 + lane] = read_i32(&ctx.banks[physical], input_base + group, lane);
        }
    }
    {
        let mut states = ball_state.lock().unwrap();
        let state = states.entry(ctx.hart_id).or_default();
        if state.chain.is_some() {
            panic!("smatmul_bias: cannot replace bias during an accumulation chain");
        }
        state.bias = Some(bias);
    };
    0
}

pub(crate) fn exec_smatmul(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let ball_state = ctx.state::<HashMap<usize, State>>();
    let m_tile = tile_rows(ctx.ball_id.expect("Ball instance"));
    let k_tile = tile_cols(ctx.ball_id.expect("Ball instance"));
    let n_tile = tile_cols(ctx.ball_id.expect("Ball instance"));
    let a_bank = rs1_b0(xs1);
    let b_bank = rs1_b1(xs1);
    let c_bank = rs1_b2(xs1);
    let rows = (xs2 & 0xfff) as usize;
    let cols = ((xs2 >> 12) & 0xfff) as usize;
    let k = rs1_iter(xs1) as usize;
    let first = (xs2 >> 24) & 1 != 0;
    let last = (xs2 >> 25) & 1 != 0;
    let output_base = ((xs2 >> 26) & 0x3f) as usize;
    if xs2 >> 32 != 0 {
        panic!("smatmul: rs2[63:32] must be zero");
    }
    if (rows != 1 && (rows == 0 || rows % m_tile != 0)) || cols != n_tile || k == 0 || k % k_tile != 0 {
        panic!("smatmul: M must be one or a positive multiple of 8, K must be a positive multiple of 16, and N must be 16");
    }
    if a_bank == b_bank || a_bank == c_bank || b_bank == c_bank {
        panic!("smatmul: A, B, and C banks must differ");
    }
    if !ctx.config(a_bank).allocated
        || !ctx.config(b_bank).allocated
        || !ctx.config(c_bank).allocated
    {
        panic!("smatmul: bank not allocated");
    }
    // A and B group g hold reduction block g (K = iter per group); C rows
    // continue across groups: line L lives in group L / bank_lines().
    let k_groups = ctx.config(a_bank).cols as usize;
    let c_groups = ctx.config(c_bank).cols as usize;
    if k_groups == 0 || ctx.config(b_bank).cols as usize != k_groups || c_groups == 0 {
        panic!("smatmul: A and B must have the same reduction groups");
    }
    if rows > (bank_lines() / 4).max(32) {
        panic!("smatmul: M exceeds the accumulator");
    }

    let a_rows = if rows == 1 { k / k_tile } else { rows * k / k_tile };
    let b_rows = k;
    let c_rows = rows * 4;
    if a_rows > bank_lines() || b_rows > bank_lines() || output_base + c_rows > c_groups * bank_lines() {
        panic!("smatmul: bank footprint exceeds bank depth");
    }

    let mut chain = {
        let mut states = ball_state.lock().unwrap();
        let state = states.entry(ctx.hart_id).or_default();
        if first {
            if state.chain.is_some() {
                panic!("smatmul: first block issued while another chain is live");
            }
            let bias = state
                .bias
                .as_ref()
                .unwrap_or_else(|| panic!("smatmul: bias must be preloaded before first block"));
            let mut accumulators = vec![0i32; rows * cols];
            for row in 0..rows {
                accumulators[row * cols..(row + 1) * cols].copy_from_slice(bias);
            }
            Chain {
                rows,
                cols,
                output_bank: c_bank,
                output_base,
                accumulators,
            }
        } else {
            state
                .chain
                .take()
                .unwrap_or_else(|| panic!("smatmul: continuation has no live chain"))
        }
    };
    if chain.rows != rows
        || chain.cols != cols
        || chain.output_bank != c_bank
        || chain.output_base != output_base
    {
        panic!("smatmul: continuation changed output shape or destination");
    }

    let k_tiles = k / k_tile;
    let row_bytes = bank_row_bytes();

    // Keep the hardware-visible operation unchanged, but traverse the data in
    // the layout used by the banks.  The old col->k loop repeatedly computed
    // addresses and reread the same B row for every output lane.  This order
    // reads one A byte and one complete B row, then updates all 16 lanes.
    // Wrapping arithmetic is intentionally retained to match the accumulator
    // semantics of the RTL and the previous emulator implementation.
    for k_group in 0..k_groups {
    let pa = pbank_group(ctx, a_bank, k_group as u64);
    let pb = pbank_group(ctx, b_bank, k_group as u64);
    for inner in 0..k {
        let b_base = ((inner / k_tile) * k_tile + inner % k_tile) * row_bytes;
        let b_row = &ctx.banks[pb][b_base..b_base + cols];
        let mut b_values = vec![0i32; n_tile];
        for col in 0..cols {
            b_values[col] = b_row[col] as i8 as i32;
        }
        for row in 0..rows {
            let a_row = if rows == 1 {
                inner / k_tile
            } else {
                ((row / m_tile) * k_tiles + inner / k_tile) * m_tile + row % m_tile
            };
            let a = ctx.banks[pa][a_row * row_bytes + inner % k_tile] as i8 as i32;
            let acc_base = row * cols;
            let acc_row = &mut chain.accumulators[acc_base..acc_base + cols];
            for col in 0..cols {
                acc_row[col] = acc_row[col].wrapping_add(a.wrapping_mul(b_values[col]));
            }
        }
    }
    }

    if last {
        for row in 0..rows {
            for col in 0..cols {
                let line = output_base + row * 4 + col / 4;
                let pc = pbank_group(ctx, c_bank, (line / bank_lines()) as u64);
                let offset = line % bank_lines() * bank_row_bytes() + col % 4 * 4;
                ctx.banks[pc][offset..offset + 4]
                    .copy_from_slice(&chain.accumulators[row * cols + col].to_le_bytes());
            }
        }
    } else {
        {
            ball_state.lock().unwrap().entry(ctx.hart_id).or_default().chain = Some(chain);
        };
    }
    0
}

pub(crate) fn bias_latency(xs1: u64, xs2: u64) -> u64 {
    let input_base = xs2 & 0x3f;
    if rs1_b1(xs1) != 0
        || rs1_b2(xs1) != 0
        || rs1_iter(xs1) != 4
        || xs2 >> 6 != 0
        || input_base + 4 > bank_lines() as u64
    {
        panic!("smatmul_bias: illegal inputBase or reserved fields");
    }
    4
}

pub(crate) fn latency(xs1: u64, xs2: u64, instance: u32) -> u64 {
    let m_tile = tile_rows(instance);
    let k_tile = tile_cols(instance);
    let n_tile = tile_cols(instance);
    let rows = xs2 & 0xfff;
    let cols = (xs2 >> 12) & 0xfff;
    let k = rs1_iter(xs1);
    let output_base = (xs2 >> 26) & 0x3f;
    if xs2 >> 32 != 0
        || (rows != 1 && (rows == 0 || rows % m_tile as u64 != 0))
        || cols != n_tile as u64
        || k == 0
        || k % k_tile as u64 != 0
    {
        panic!("smatmul: illegal matrix encoding");
    }
    rows * cols * k / m_tile as u64 + if (xs2 >> 25) & 1 != 0 { rows * 4 } else { 0 }
}
