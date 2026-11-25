pub fn q8(bits: u32) -> i32 {
    let value = f32::from_bits(bits);
    assert!(!value.is_nan());
    ((value as f64).clamp(-4096.0, 4096.0) * 256.0).round_ties_even() as i32
}

pub fn expected(input: &[u32], table: &[u32], rows: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    let stride = k.div_ceil(4) * 4;
    let mut output = vec![0x5a5a5a5a; 4096];
    let mut lookups = Vec::new();
    for row in 0..rows {
        let scores: Vec<i32> = input[row * stride..row * stride + k].iter().map(|&v| q8(v)).collect();
        let maximum = *scores.iter().max().unwrap();
        let indices: Vec<usize> = scores.iter().map(|&v| (maximum-v).min(4095) as usize).collect();
        let sum: u64 = indices.iter().map(|&i| table[i] as u64).sum();
        output[row * stride..(row + 1) * stride].fill(0);
        for (col, index) in indices.into_iter().enumerate() {
            let q = (table[index] as u64 * 32768 + sum / 2) / sum;
            output[row * stride + col] = (q as f32 / 32768.0).to_bits();
            lookups.push(index as u32);
        }
    }
    (output, lookups)
}
