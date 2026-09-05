pub const BANK_LINES: usize = 64;
pub const BANK_ROW_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geo {
    pub input_side: usize,
    pub output_side: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub input_base: usize,
    pub output_base: usize,
    pub output_stride: usize,
    pub start_row: usize,
    pub start_col: usize,
}

pub fn validate(geo: &Geo) {
    let g = geo;
    let dimensions_valid =
        g.input_side != 0 && g.output_side != 0 && g.kernel != 0 && g.stride != 0;
    if !dimensions_valid {
        panic!("maxpool: illegal square pooling geometry");
    }
    let last_row = g.start_row + (g.output_side - 1) * g.stride + g.kernel - 1;
    let last_col = g.start_col + (g.output_side - 1) * g.stride + g.kernel - 1;
    let input_end = g.padding + g.input_side;
    let reads_input = g.start_row < input_end
        && g.start_col < input_end
        && last_row >= g.padding
        && last_col >= g.padding;
    let input_fits = !reads_input
        || g.input_base
            + (last_row.min(input_end - 1) - g.padding) * g.input_side
            + (last_col.min(input_end - 1) - g.padding)
            < BANK_LINES;
    if g.output_stride < g.output_side
        || !input_fits
        || g.output_base + (g.output_side - 1) * g.output_stride + g.output_side > BANK_LINES
        || g.input_side + 2 * g.padding < g.kernel + g.start_row
        || g.input_side + 2 * g.padding < g.kernel + g.start_col
        || g.start_row + (g.output_side - 1) * g.stride + g.kernel > g.input_side + 2 * g.padding
        || g.start_col + (g.output_side - 1) * g.stride + g.kernel > g.input_side + 2 * g.padding
    {
        panic!("maxpool: illegal square pooling geometry");
    }
}

pub fn iter_for(geo: &Geo) -> u32 {
    let iter = geo.output_side * geo.output_side;
    if iter == 0 {
        panic!("maxpool: iter must be positive");
    }
    iter as u32
}

pub fn maxpool_rows(input: &[u8], geo: &Geo) -> Vec<(u32, [u8; BANK_ROW_BYTES])> {
    validate(geo);
    let tile = geo.input_side * geo.input_side;
    if input.len() != tile * BANK_ROW_BYTES {
        panic!("maxpool: input bytes {} != tile {} rows", input.len(), tile);
    }

    let mut out = Vec::with_capacity(geo.output_side * geo.output_side);
    for output_y in 0..geo.output_side {
        for output_x in 0..geo.output_side {
            let mut maximum = [0x80u8; BANK_ROW_BYTES];
            for kernel_y in 0..geo.kernel {
                for kernel_x in 0..geo.kernel {
                    let input_y = output_y * geo.stride + kernel_y + geo.start_row;
                    let input_x = output_x * geo.stride + kernel_x + geo.start_col;
                    let sy = input_y as isize - geo.padding as isize;
                    let sx = input_x as isize - geo.padding as isize;
                    if sy < 0
                        || sx < 0
                        || sy >= geo.input_side as isize
                        || sx >= geo.input_side as isize
                    {
                        continue;
                    }
                    let row = (sy as usize) * geo.input_side + sx as usize;
                    let off = row * BANK_ROW_BYTES;
                    for lane in 0..BANK_ROW_BYTES {
                        let v = input[off + lane] as i8;
                        if v > maximum[lane] as i8 {
                            maximum[lane] = v as u8;
                        }
                    }
                }
            }
            let addr = (geo.output_base + output_y * geo.output_stride + output_x) as u32;
            out.push((addr, maximum));
        }
    }
    out
}

pub fn pack_row(bytes: &[u8; BANK_ROW_BYTES]) -> u128 {
    let mut word = 0u128;
    for (lane, byte) in bytes.iter().enumerate() {
        word |= u128::from(*byte) << (lane * 8);
    }
    word
}
