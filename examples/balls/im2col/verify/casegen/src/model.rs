pub const TILE: usize = 16;
pub const BANK_ROW_BYTES: usize = TILE;

pub fn output_rows(iter: usize, ksize: usize, stride: usize, padding: usize) -> usize {
    if iter == 0 || ksize == 0 || stride == 0 {
        panic!("im2col model: iter/ksize/stride must be >= 1");
    }
    let padded = iter + 2 * padding;
    if padded < ksize {
        panic!("im2col model: kernel larger than padded input");
    }
    let out_dim = (padded - ksize) / stride + 1;
    let windows = out_dim * out_dim;
    let kernel = ksize * ksize;
    let m_tiles = windows.div_ceil(TILE);
    let k_tiles = kernel.div_ceil(TILE);
    m_tiles * k_tiles * TILE
}

pub fn im2col(
    src: &[u8],
    iter: usize,
    ksize: usize,
    stride: usize,
    padding: usize,
) -> Vec<u8> {
    if src.len() != iter * iter {
        panic!(
            "im2col model: src len {} != iter*iter {}",
            src.len(),
            iter * iter
        );
    }
    let padded = iter + 2 * padding;
    if padded < ksize {
        panic!("im2col model: kernel larger than padded input");
    }
    if stride == 0 {
        panic!("im2col model: stride must be >= 1");
    }
    let out_dim = (padded - ksize) / stride + 1;
    let windows = out_dim * out_dim;
    let kernel_elems = ksize * ksize;
    let m_tiles = windows.div_ceil(TILE);
    let k_tiles = kernel_elems.div_ceil(TILE);
    let rows = m_tiles * k_tiles * TILE;
    let mut output = vec![0u8; rows * TILE];

    let mut window = 0usize;
    for orow in 0..out_dim {
        for ocol in 0..out_dim {
            for kr in 0..ksize {
                for kc in 0..ksize {
                    let kernel = kr * ksize + kc;
                    let m_tile = window / TILE;
                    let m_row = window % TILE;
                    let k_tile = kernel / TILE;
                    let lane = kernel % TILE;
                    let bank_row = (m_tile * k_tiles + k_tile) * TILE + m_row;
                    let out = bank_row * TILE + lane;
                    if out >= output.len() {
                        panic!("im2col model: output range out={out}");
                    }
                    let input_row = (orow * stride + kr) as isize - padding as isize;
                    let input_col = (ocol * stride + kc) as isize - padding as isize;
                    if input_row >= 0
                        && input_row < iter as isize
                        && input_col >= 0
                        && input_col < iter as isize
                    {
                        let src_i = input_row as usize * iter + input_col as usize;
                        output[out] = src[src_i];
                    }
                }
            }
            window += 1;
        }
    }
    output
}

pub fn pack_bank_words(flat: &[u8]) -> Vec<u8> {
    let nwords = flat.len().div_ceil(BANK_ROW_BYTES);
    let mut out = vec![0u8; nwords * BANK_ROW_BYTES];
    out[..flat.len()].copy_from_slice(flat);
    out
}

pub fn num_words(total_bytes: usize) -> usize {
    if total_bytes % BANK_ROW_BYTES != 0 {
        panic!("num_words: total_bytes {total_bytes} not a multiple of bank row");
    }
    total_bytes / BANK_ROW_BYTES
}
