use super::decode::{pbank, pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use super::instruction::ExecContext;
use bebop_bemu::ball::blink::{bank_lines, bank_row_bytes};
use std::collections::HashMap;

fn tile_shape(instance: u32) -> (usize, usize) {
    let p = bebop_bemu::ball::blink::parameters::<super::params::Params>("smatmul", instance);
    (
        p.tile_rows.expect("tileRows") as usize,
        p.tile_cols.expect("tileCols") as usize,
    )
}

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
    assert!(
        !rs1_read_old(xs1),
        "smatmul_bias: reads no old destination; rs1[29] must be clear"
    );
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
    let mut bias = vec![0i32; tile_shape(ctx.ball_id.expect("Ball instance")).1];
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

/// Sixteen signed output columns. VNNI groups four products into each wrapping i32 lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni")]
unsafe fn dot16(
    a: &[u8],
    b: &[u8],
    sums: &mut [i32],
    k: usize,
    tile_rows: usize,
    row_bytes: usize,
) {
    use std::arch::x86_64::*;
    let dot = |mut acc: __m512i, a: __m512i, b: __m512i| {
        std::arch::asm!(
            "vpdpbusd {acc}, {a}, {b}",
            acc = inout(zmm_reg) acc, a = in(zmm_reg) a, b = in(zmm_reg) b,
            options(pure, nomem, nostack),
        );
        acc
    };
    let mut packed = Vec::with_capacity(k / 4);
    let mut bias = _mm512_setzero_si512();
    for group in 0..k / 4 {
        let rows = &b[group * 4 * row_bytes..];
        let bytes: [[u8; 4]; 16] = std::array::from_fn(|col| {
            [
                rows[col],
                rows[row_bytes + col],
                rows[2 * row_bytes + col],
                rows[3 * row_bytes + col],
            ]
        });
        let w = _mm512_loadu_si512(bytes.as_ptr().cast());
        // VNNI's first input is unsigned. Removing 128 * sum(weights) restores signed A.
        bias = dot(bias, _mm512_set1_epi8(-128), w);
        packed.push(w);
    }
    let stride = if sums.len() == 16 {
        row_bytes
    } else {
        tile_rows * row_bytes
    };
    for (row, lanes) in sums.chunks_exact_mut(16).enumerate() {
        let base = (row / tile_rows * (k / 16) * tile_rows + row % tile_rows) * row_bytes;
        let mut acc = _mm512_sub_epi32(_mm512_loadu_si512(lanes.as_ptr().cast()), bias);
        for (tile, weights) in packed.chunks_exact(4).enumerate() {
            let start = base + tile * stride;
            for (bytes, &w) in a[start..start + 16].chunks_exact(4).zip(weights) {
                let x =
                    i32::from_le_bytes(bytes.try_into().unwrap()) ^ i32::from_ne_bytes([0x80; 4]);
                acc = dot(acc, _mm512_set1_epi32(x), w);
            }
        }
        _mm512_storeu_si512(lanes.as_mut_ptr().cast(), acc);
    }
}

pub(crate) fn exec_smatmul(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    assert!(
        !rs1_read_old(xs1),
        "smatmul: reads no old destination; rs1[29] must be clear"
    );
    let ball_state = ctx.state::<HashMap<usize, State>>();
    let (m_tile, k_tile) = tile_shape(ctx.ball_id.expect("Ball instance"));
    let n_tile = k_tile;
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
    if (rows != 1 && (rows == 0 || rows % m_tile != 0))
        || cols != n_tile
        || k == 0
        || k % k_tile != 0
    {
        panic!(
            "smatmul: M must be one or a positive multiple of tileRows, K must be a positive multiple of 16, and N must be 16"
        );
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

    let a_rows = if rows == 1 {
        k / k_tile
    } else {
        rows * k / k_tile
    };
    let b_rows = k;
    let c_rows = rows * 4;
    if a_rows > bank_lines()
        || b_rows > bank_lines()
        || output_base + c_rows > c_groups * bank_lines()
    {
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

    // Keep a block of output lanes in host registers across the reduction.
    const LANES: usize = 16;
    let a_tile_stride = if rows == 1 {
        row_bytes
    } else {
        m_tile * row_bytes
    };

    for k_group in 0..k_groups {
        let pa = pbank_group(ctx, a_bank, k_group as u64);
        let pb = pbank_group(ctx, b_bank, k_group as u64);
        let a_data = &ctx.banks[pa];
        let b_data = &ctx.banks[pb];
        #[cfg(target_arch = "x86_64")]
        if cols == 16 && k_tile == 16 && std::is_x86_feature_detected!("avx512vnni") {
            unsafe {
                dot16(
                    a_data,
                    b_data,
                    &mut chain.accumulators,
                    k,
                    m_tile,
                    row_bytes,
                )
            };
            continue;
        }
        let mut weights = vec![[0i32; LANES]; k];
        for col in (0..cols).step_by(LANES) {
            let width = (cols - col).min(LANES);
            for (inner, expanded) in weights.iter_mut().enumerate() {
                let offset = inner * row_bytes + col;
                for (dst, &byte) in expanded.iter_mut().zip(&b_data[offset..offset + width]) {
                    *dst = byte as i8 as i32;
                }
            }
            for (row, acc) in chain.accumulators.chunks_exact_mut(cols).enumerate() {
                let base = (row / m_tile * k_tiles * m_tile + row % m_tile) * row_bytes;
                let mut sums = [0i32; LANES];
                sums[..width].copy_from_slice(&acc[col..col + width]);
                for (tile, panel) in weights.chunks_exact(k_tile).enumerate() {
                    let start = base + tile * a_tile_stride;
                    for (&byte, weights) in a_data[start..start + k_tile].iter().zip(panel) {
                        let a = byte as i8 as i32;
                        for (sum, &b) in sums.iter_mut().zip(weights) {
                            *sum = sum.wrapping_add(a.wrapping_mul(b));
                        }
                    }
                }
                acc[col..col + width].copy_from_slice(&sums[..width]);
            }
        }
    }

    if last {
        let lines = bank_lines();
        for (row, accumulators) in chain.accumulators.chunks_exact(cols).enumerate() {
            for (group, values) in accumulators.chunks(4).enumerate() {
                let line = output_base + row * 4 + group;
                let pc = pbank_group(ctx, c_bank, (line / lines) as u64);
                let offset = line % lines * row_bytes;
                let dest = &mut ctx.banks[pc][offset..offset + values.len() * 4];
                for (word, value) in dest.chunks_exact_mut(4).zip(values) {
                    word.copy_from_slice(&value.to_le_bytes());
                }
            }
        }
    } else {
        {
            ball_state
                .lock()
                .unwrap()
                .entry(ctx.hart_id)
                .or_default()
                .chain = Some(chain);
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
    super::latency::bias_cycles()
}

pub(crate) fn os_latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let (rows, cols) = tile_shape(ctx.ball_id.expect("Ball instance"));
    super::latency::os_cycles(
        rs1_iter(xs1),
        xs2,
        ctx.config(rs1_b0(xs1)).cols,
        rows as u64,
        cols as u64,
    )
}
