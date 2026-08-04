pub const BANK_ROW_BYTES: usize = 16;

pub fn transpose_bytes(src: &[u8], iter: usize, w: usize, elem_bytes: usize) -> Vec<u8> {
    let total = iter * w;
    assert_eq!(src.len(), total * elem_bytes);
    let mut dst = vec![0u8; src.len()];
    for r in 0..iter {
        for c in 0..w {
            let s = (r * w + c) * elem_bytes;
            let d = (c * iter + r) * elem_bytes;
            dst[d..d + elem_bytes].copy_from_slice(&src[s..s + elem_bytes]);
        }
    }
    dst
}

pub fn num_words(total_bytes: usize) -> usize {
    assert_eq!(total_bytes % BANK_ROW_BYTES, 0, "total_bytes not a multiple of bank row");
    total_bytes / BANK_ROW_BYTES
}
