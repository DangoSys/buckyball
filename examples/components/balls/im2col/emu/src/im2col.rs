use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;
use bebop_bemu::thread::banks::{bank_lines, bank_row_bytes};

fn max_iter(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<
        bebop_bemu::configs::ball_domain::parameters::im2col::Params,
    >("im2col", instance)
    .max_iter
    .expect("maxIter") as usize
}

fn max_kernel(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<
        bebop_bemu::configs::ball_domain::parameters::im2col::Params,
    >("im2col", instance)
    .max_k_size
    .expect("maxKSize") as usize
}

fn max_padding(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<
        bebop_bemu::configs::ball_domain::parameters::im2col::Params,
    >("im2col", instance)
    .max_padding
    .expect("maxPadding") as usize
}

#[derive(Clone, Copy)]
struct Shape {
    input_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    start_col: usize,
    start_row: usize,
    input_base: usize,
    window_start: usize,
    window_count: usize,
    tile_rows: usize,
}

fn decode_shape(xs1: u64, xs2: u64) -> Shape {
    if rs1_b1(xs1) != 0 {
        panic!("im2col: input bank 1 must be zero");
    }
    if xs2 >> 51 != 0 {
        panic!("im2col: rs2[63:51] must be zero");
    }
    Shape {
        input_size: rs1_iter(xs1) as usize,
        kernel: (xs2 & 0xf) as usize,
        stride: ((xs2 >> 4) & 0xf) as usize,
        padding: ((xs2 >> 8) & 0xf) as usize,
        start_col: ((xs2 >> 12) & 0xf) as usize,
        start_row: ((xs2 >> 16) & 0xf) as usize,
        input_base: ((xs2 >> 20) & 0x3f) as usize,
        window_start: ((xs2 >> 26) & 0x3ff) as usize,
        window_count: ((xs2 >> 36) & 0x3ff) as usize,
        tile_rows: ((xs2 >> 46) & 0x1f) as usize,
    }
}

// Returns the output columns of the full convolution output.
fn output_cols(shape: Shape, instance: u32) -> usize {
    if shape.input_size == 0
        || shape.input_size > max_iter(instance)
        || shape.kernel == 0
        || shape.kernel > max_kernel(instance)
        || shape.stride == 0
        || shape.padding > max_padding(instance)
        || shape.start_row > shape.padding
        || shape.start_col > shape.padding
        || shape.window_count == 0
        || !shape.tile_rows.is_power_of_two()
        || shape.tile_rows > 16
        || shape.input_base + shape.input_size * shape.input_size > bank_lines()
    {
        panic!("im2col: illegal shape field");
    }
    let padded = shape.input_size + shape.padding * 2;
    if padded < shape.kernel + shape.start_row || padded < shape.kernel + shape.start_col {
        panic!("im2col: kernel and start exceed padded input");
    }
    let row_numerator = padded - shape.kernel - shape.start_row;
    let col_numerator = padded - shape.kernel - shape.start_col;
    if row_numerator % shape.stride != 0 || col_numerator % shape.stride != 0 {
        panic!("im2col: output shape is not integral");
    }
    let output_rows = row_numerator / shape.stride + 1;
    let output_cols = col_numerator / shape.stride + 1;
    if shape.window_start + shape.window_count > output_rows * output_cols {
        panic!("im2col: selected window range exceeds output shape");
    }
    output_cols
}

pub(crate) fn exec_im2col(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input_bank = rs1_b0(xs1);
    let output_bank = rs1_b2(xs1);
    if input_bank == output_bank {
        panic!("im2col: input and output banks must differ");
    }
    if !ctx.config(input_bank).allocated || !ctx.config(output_bank).allocated {
        panic!("im2col: bank not allocated");
    }
    let shape = decode_shape(xs1, xs2);
    let output_cols = output_cols(shape, ctx.ball_id.expect("Ball instance"));

    // Input group g is channel block g. K is ordered (kernel row, kernel
    // column, channel), so K tile t = kernel_position * channel_groups + g is
    // one whole input row. Output group j holds K tiles j*group_tiles.. in the
    // SMatMul A layout with tile_rows-row M tiles.
    let channel_groups = ctx.config(input_bank).cols as usize;
    let output_groups = ctx.config(output_bank).cols as usize;
    let k_tiles = shape.kernel * shape.kernel * channel_groups;
    if channel_groups == 0 || output_groups == 0 || k_tiles % output_groups != 0 {
        panic!("im2col: output groups must divide the K tiles");
    }
    let group_tiles = k_tiles / output_groups;
    let padded_windows = shape.window_count.div_ceil(shape.tile_rows) * shape.tile_rows;
    if padded_windows * group_tiles > bank_lines() {
        panic!("im2col: output footprint exceeds bank depth");
    }

    let row_bytes = bank_row_bytes();
    for tile in 0..k_tiles {
        let channel_group = tile % channel_groups;
        let kernel_row = tile / channel_groups / shape.kernel;
        let kernel_col = tile / channel_groups % shape.kernel;
        let pi = pbank_group(ctx, input_bank, channel_group as u64);
        let po = pbank_group(ctx, output_bank, (tile / group_tiles) as u64);
        let (input, output) = ctx.banks.read_write(pi, po);
        for window in 0..padded_windows {
            let line = ((window / shape.tile_rows) * group_tiles + tile % group_tiles)
                * shape.tile_rows
                + window % shape.tile_rows;
            let target = &mut output[line * row_bytes..(line + 1) * row_bytes];
            target.fill(0);
            if window >= shape.window_count {
                continue;
            }
            let global = shape.window_start + window;
            let source_row = (shape.start_row + global / output_cols * shape.stride + kernel_row)
                as isize
                - shape.padding as isize;
            let source_col = (shape.start_col + global % output_cols * shape.stride + kernel_col)
                as isize
                - shape.padding as isize;
            if source_row < 0
                || source_col < 0
                || source_row >= shape.input_size as isize
                || source_col >= shape.input_size as isize
            {
                continue;
            }
            let source =
                shape.input_base + source_row as usize * shape.input_size + source_col as usize;
            target.copy_from_slice(&input[source * row_bytes..(source + 1) * row_bytes]);
        }
    }
    0
}

pub(crate) fn latency(xs1: u64, xs2: u64, instance: u32) -> u64 {
    let shape = decode_shape(xs1, xs2);
    output_cols(shape, instance);
    (shape.window_count.div_ceil(shape.tile_rows)
        * shape.tile_rows
        * shape.kernel
        * shape.kernel
        * 2) as u64
}
