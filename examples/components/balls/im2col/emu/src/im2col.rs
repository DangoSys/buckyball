use bebop_bemu::thread::banks::{bank_lines, bank_row_bytes};
use super::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use super::instruction::ExecContext;

const CORE: usize = 16;

fn max_iter(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::im2col::Params>("im2col", instance).max_iter.expect("maxIter") as usize
}

fn max_kernel(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::im2col::Params>("im2col", instance).max_k_size.expect("maxKSize") as usize
}

fn max_padding(instance: u32) -> usize {
    bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::im2col::Params>("im2col", instance).max_padding.expect("maxPadding") as usize
}

#[derive(Clone, Copy)]
struct Shape {
    input_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    start_row: usize,
    start_col: usize,
    input_base: usize,
    lane: usize,
    window_start: usize,
    window_count: usize,
}

fn decode_shape(xs1: u64, xs2: u64) -> Shape {
    Shape {
        input_size: rs1_iter(xs1) as usize,
        kernel: (xs2 & 0xff) as usize,
        stride: ((xs2 >> 8) & 0xff) as usize,
        padding: ((xs2 >> 16) & 0xff) as usize,
        start_col: ((xs2 >> 24) & 0xff) as usize,
        start_row: ((xs2 >> 32) & 0xff) as usize,
        input_base: ((xs2 >> 40) & 0x3f) as usize,
        lane: ((xs2 >> 46) & 0xf) as usize,
        window_start: ((xs2 >> 50) & 0x3f) as usize,
        window_count: ((xs2 >> 56) & 0x7f) as usize,
    }
}

fn dimensions(shape: Shape, instance: u32) -> (usize, usize, usize) {
    if shape.input_size == 0
        || shape.input_size > max_iter(instance)
        || shape.kernel == 0
        || shape.kernel > max_kernel(instance)
        || shape.stride == 0
        || shape.padding > max_padding(instance)
        || shape.start_row > shape.padding
        || shape.start_col > shape.padding
        || shape.window_count == 0
        || shape.window_start >= 64
        || shape.window_count > 64
    {
        panic!("im2col: illegal shape field");
    }
    let padded = shape
        .input_size
        .checked_add(shape.padding * 2)
        .unwrap_or_else(|| panic!("im2col: padded input overflow"));
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
    let windows = output_rows
        .checked_mul(output_cols)
        .unwrap_or_else(|| panic!("im2col: output window count overflow"));
    if shape.window_start >= windows || shape.window_start + shape.window_count > windows {
        panic!("im2col: selected window range exceeds output shape");
    }
    (output_rows, output_cols, windows)
}

fn footprints(shape: Shape) -> (usize, usize) {
    let input_values = shape
        .input_size
        .checked_mul(shape.input_size)
        .unwrap_or_else(|| panic!("im2col: input footprint overflow"));
    let kernel_values = shape.kernel * shape.kernel;
    let input_rows = input_values;
    let output_rows = shape.window_count.div_ceil(CORE) * kernel_values.div_ceil(CORE) * CORE;
    if shape.input_base + input_rows > bank_lines() || output_rows > bank_lines() {
        panic!("im2col: input or output footprint exceeds bank depth");
    }
    (input_rows, output_rows)
}

pub(crate) fn exec_im2col(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input_bank = rs1_b0(xs1);
    let output_bank = rs1_b2(xs1);
    if rs1_b1(xs1) != 0 {
        panic!("im2col: input bank 1 must be zero");
    }
    if xs2 >> 63 != 0 {
        panic!("im2col: rs2[63] must be zero");
    }
    if input_bank == output_bank {
        panic!("im2col: input and output banks must differ");
    }
    if !ctx.config(input_bank).allocated || !ctx.config(output_bank).allocated {
        panic!("im2col: bank not allocated");
    }
    if ctx.config(input_bank).cols != 1 || ctx.config(output_bank).cols != 1 {
        panic!("im2col: input and output banks must each have one column");
    }

    let shape = decode_shape(xs1, xs2);
    let (_, output_cols, _) = dimensions(shape, ctx.ball_id.expect("Ball instance"));
    let (_, output_bank_rows) = footprints(shape);
    let kernel_values = shape.kernel * shape.kernel;
    let kernel_tiles = kernel_values.div_ceil(CORE);
    let pi = pbank(ctx, input_bank);
    let po = pbank(ctx, output_bank);
    let (input, output) = ctx.banks.read_write(pi, po);
    output[..output_bank_rows * bank_row_bytes()].fill(0);

    for local_window in 0..shape.window_count {
        let global_window = shape.window_start + local_window;
        let output_row = global_window / output_cols;
        let output_col = global_window % output_cols;
        for kernel_row in 0..shape.kernel {
            for kernel_col in 0..shape.kernel {
                let source_row = (shape.start_row + output_row * shape.stride + kernel_row)
                    as isize
                    - shape.padding as isize;
                let source_col = (shape.start_col + output_col * shape.stride + kernel_col)
                    as isize
                    - shape.padding as isize;
                if source_row < 0
                    || source_col < 0
                    || source_row >= shape.input_size as isize
                    || source_col >= shape.input_size as isize
                {
                    continue;
                }
                let kernel_element = kernel_row * shape.kernel + kernel_col;
                let bank_row = ((local_window / CORE) * kernel_tiles + kernel_element / CORE)
                    * CORE
                    + local_window % CORE;
                let lane = kernel_element % CORE;
                let source = source_row as usize * shape.input_size + source_col as usize;
                output[bank_row * CORE + lane] =
                    input[(shape.input_base + source) * CORE + shape.lane];
            }
        }
    }
    0
}

pub(crate) fn latency(xs1: u64, xs2: u64, instance: u32) -> u64 {
    if rs1_b1(xs1) != 0 {
        panic!("im2col: input bank 1 must be zero");
    }
    if xs2 >> 63 != 0 {
        panic!("im2col: rs2[63] must be zero");
    }
    let shape = decode_shape(xs1, xs2);
    dimensions(shape, instance);
    footprints(shape);
    (shape.window_count * shape.kernel * shape.kernel).max(16) as u64
}
