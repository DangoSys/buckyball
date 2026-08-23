use super::super::bank::{bank_lines, bank_num};
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

/// Must match examples/balls/im2col/configs/default.toml
const MAX_ITER: usize = 34;
const MAX_KSIZE: usize = 7;
const MAX_PADDING: usize = 7;
const TILE: usize = 16;

#[derive(Clone, Copy)]
struct Shape {
    iter: usize,
    ksize: usize,
    stride: usize,
    padding: usize,
    start_row: usize,
    start_col: usize,
}

fn decode_shape(xs1: u64, xs2: u64) -> Shape {
    Shape {
        iter: rs1_iter(xs1) as usize,
        ksize: (xs2 & 0xff) as usize,
        stride: ((xs2 >> 8) & 0xff) as usize,
        padding: ((xs2 >> 16) & 0xff) as usize,
        start_col: ((xs2 >> 24) & 0xff) as usize,
        start_row: ((xs2 >> 32) & 0xff) as usize,
    }
}

pub(crate) fn exec_im2col(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let op1 = rs1_b0(xs1);
    let wr = rs1_b2(xs1);

    if op1 >= bank_num() as u64 || wr >= bank_num() as u64 {
        panic!("im2col: invalid bank_id");
    }
    if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
        panic!("im2col: bank not allocated");
    }
    if op1 == wr {
        panic!("im2col: op1 and wr must differ");
    }

    let Shape {
        iter,
        ksize,
        stride,
        padding,
        start_row,
        start_col,
    } = decode_shape(xs1, xs2);

    if iter == 0 || iter > MAX_ITER {
        panic!("im2col: iter out of range 1..={MAX_ITER} (got {iter})");
    }
    if ksize == 0 || ksize > MAX_KSIZE {
        panic!("im2col: ksize out of range 1..={MAX_KSIZE} (got {ksize})");
    }
    if stride == 0 {
        panic!("im2col: stride must be >= 1");
    }
    if padding > MAX_PADDING {
        panic!("im2col: padding out of range 0..={MAX_PADDING} (got {padding})");
    }
    if start_row > padding || start_col > padding {
        panic!(
            "im2col: start_row/col out of range 0..=padding \
                 (start_row={start_row}, start_col={start_col}, padding={padding})"
        );
    }
    let padded_size = iter + 2 * padding;
    if padded_size < ksize + start_row || padded_size < ksize + start_col {
        panic!("im2col: kernel+start larger than padded input");
    }
    if (padded_size - ksize - start_row) % stride != 0
        || (padded_size - ksize - start_col) % stride != 0
    {
        panic!("im2col: padded/ksize/start/stride yield non-integer output");
    }

    let output_rows_n = (padded_size - ksize - start_row) / stride + 1;
    let output_cols_n = (padded_size - ksize - start_col) / stride + 1;
    if output_rows_n != output_cols_n {
        panic!("im2col: non-square output ({output_rows_n}x{output_cols_n}) unsupported");
    }
    let output_dim = output_rows_n;
    let windows = output_dim * output_dim;
    let kernel_elems = ksize * ksize;
    let m_tiles = windows.div_ceil(TILE);
    let k_tiles = kernel_elems.div_ceil(TILE);
    let output_rows = m_tiles * k_tiles * TILE;

    let groups = ctx.cfgs[wr as usize].cols as usize;
    let rows_per_bank = bank_lines();
    let capacity_rows = groups * rows_per_bank;
    if output_rows > capacity_rows {
        panic!(
            "im2col: outputRows={output_rows} exceeds bank capacity={capacity_rows} \
                 (groups={groups}, bank_lines={rows_per_bank})"
        );
    }

    let po = pbank(ctx.bank_map, op1);
    let mut output = vec![0u8; output_rows * TILE];

    let srcb = &ctx.banks[po];
    let mut window = 0usize;
    for output_row in 0..output_dim {
        for output_col in 0..output_dim {
            for kr in 0..ksize {
                for kc in 0..ksize {
                    let input_row =
                        (start_row + output_row * stride + kr) as isize - padding as isize;
                    let input_col =
                        (start_col + output_col * stride + kc) as isize - padding as isize;
                    let kernel = kr * ksize + kc;
                    let m_tile = window / TILE;
                    let m_row = window % TILE;
                    let k_tile = kernel / TILE;
                    let lane = kernel % TILE;
                    let bank_row = (m_tile * k_tiles + k_tile) * TILE + m_row;
                    let out = bank_row * TILE + lane;
                    if out >= output.len() {
                        panic!("im2col: output range out={out}");
                    }
                    if input_row >= 0
                        && input_row < iter as isize
                        && input_col >= 0
                        && input_col < iter as isize
                    {
                        let src = input_row as usize * iter + input_col as usize;
                        if src >= srcb.len() {
                            panic!("im2col: input range src={src}");
                        }
                        output[out] = srcb[src];
                    }
                }
            }
            window += 1;
        }
    }

    for (row, data) in output.chunks_exact(TILE).enumerate() {
        let group = row / rows_per_bank;
        let local_row = row % rows_per_bank;
        if group >= groups {
            panic!("im2col: output requires group {group}, allocated groups={groups}");
        }
        let pw = pbank_group(ctx.bank_map, wr, group as u64);
        let off = local_row * TILE;
        ctx.banks[pw][off..off + TILE].copy_from_slice(data);
    }
    0
}

pub(crate) fn latency(xs1: u64, xs2: u64) -> u64 {
    let Shape {
        iter,
        ksize,
        stride,
        padding,
        start_row,
        start_col,
    } = decode_shape(xs1, xs2);
    let iter = iter as u64;
    let ksize = ksize as u64;
    let stride = stride as u64;
    let padding = padding as u64;
    let start_row = start_row as u64;
    let start_col = start_col as u64;

    if iter == 0
        || iter > MAX_ITER as u64
        || ksize == 0
        || ksize > MAX_KSIZE as u64
        || stride == 0
        || padding > MAX_PADDING as u64
        || start_row > padding
        || start_col > padding
    {
        return 16;
    }
    let padded_size = iter + 2 * padding;
    if padded_size < ksize + start_row || padded_size < ksize + start_col {
        return 16;
    }
    if (padded_size - ksize - start_row) % stride != 0
        || (padded_size - ksize - start_col) % stride != 0
    {
        return 16;
    }

    let output_rows_n = (padded_size - ksize - start_row) / stride + 1;
    let output_cols_n = (padded_size - ksize - start_col) / stride + 1;
    output_rows_n
        .saturating_mul(output_cols_n)
        .saturating_mul(ksize)
        .saturating_mul(ksize)
        .max(16)
}
