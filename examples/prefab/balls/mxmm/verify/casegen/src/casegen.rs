// MXMM reference model, the contract shared with the BEMU model
// (emu/src/mxfp8.rs, emu/src/fp32.rs) and MxmmBall:
//
// rs2[11:0] = M (1 or a multiple of 16), rs2[23:12] = N (multiple of 16),
// rs2[24] = first, rs2[25] = last, rs2[31:26] = C base row, iter = K.
// MXMM_MXFP8_WINDOW adds rs2[47:32] = fullK and rs2[63:48] = startK; the other
// formats reserve rs2[63:32].
//
// MXFP8 (K multiple of 32): A holds M rows of fullK E4M3 codes followed by the
// E8M0 scales (row r, block j at byte M * fullK + r * fullK / 32 + j); B holds
// N columns of K codes followed by their scales. FP32 (K multiple of four): A
// is M rows of K floats, B is N columns of K floats.
// C[r][c] accumulates sum_k A[r][k] * B[c][k] in k order with a fused
// multiply-add (MXFP8, FMA32) or a rounded product then a rounded add (F32),
// starting from zero on first and from the chain otherwise. On last the chain
// is written to C as row-major FP32 from row base; otherwise nothing is
// written.

pub const ROW_BYTES: usize = 16;
pub const NUM_CASES: u32 = 8;

pub const FORMAT_MXFP8: u32 = 0;
pub const FORMAT_WINDOW: u32 = 1;
pub const FORMAT_FMA32: u32 = 2;
pub const FORMAT_F32: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MxmmCmd {
    pub format: u32,
    pub rob_id: u32,
    pub iter: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub op2_col: u32,
    pub wr_col: u32,
}

#[derive(Clone, Debug)]
pub struct Preload {
    pub bank: u32,
    pub group: u32,
    pub addr: u32,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Write {
    pub addr: u32,
    pub mask: u16,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub cmd: MxmmCmd,
    pub preload: Vec<Preload>,
    pub reads: [Vec<u32>; 2],
    pub writes: Vec<Write>,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub steps: Vec<Step>,
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    // E4M3 code other than NaN (0x7f, 0xff).
    fn code(&mut self) -> u8 {
        loop {
            let code = self.next() as u8;
            if code & 0x7f != 0x7f {
                return code;
            }
        }
    }

    // E8M0 scale near one, sometimes the smallest scale.
    fn scale(&mut self) -> u8 {
        if self.next() % 8 == 0 {
            0
        } else {
            120 + (self.next() % 14) as u8
        }
    }

    fn float(&mut self) -> f32 {
        let bits = (self.next() & 0x807f_ffff) | ((120 + self.next() % 12) << 23);
        f32::from_bits(bits)
    }
}

fn word(bytes: &[u8]) -> u128 {
    let mut out = 0u128;
    for (lane, byte) in bytes.iter().enumerate() {
        out |= u128::from(*byte) << (lane * 8);
    }
    out
}

fn decode(code: u8, scale: u8) -> f32 {
    assert!(code & 0x7f != 0x7f && scale != 255, "mxfp8: NaN encoding");
    let factor = if scale == 0 { f32::from_bits(1 << 22) } else { f32::from_bits((scale as u32) << 23) };
    let exponent = (code >> 3) & 15;
    let mantissa = (code & 7) as u32;
    let normal = ((exponent as u32 + 120) << 23) | (mantissa << 20);
    let subnormal = (mantissa as f32 / 512.0).to_bits();
    let element = f32::from_bits(if exponent == 0 { subnormal } else { normal });
    let value = f32::from_bits((element * factor).to_bits() | ((code as u32 & 128) << 24));
    assert!(value.is_finite(), "mxfp8: decoded value overflows FP32");
    value
}

// One operand bank: its byte image and the rows the Ball reads, in order.
struct Operand {
    bytes: Vec<u8>,
    reads: Vec<u32>,
    // values[row][k] (A) or values[col][k] (B) seen by the command.
    values: Vec<Vec<f32>>,
}

fn sequential_reads(words: usize, start: usize) -> Vec<u32> {
    (start..start + words).map(|w| w as u32).collect()
}

fn mx_operand(rng: &mut Rng, rows: usize, k: usize) -> Operand {
    let codes: Vec<u8> = (0..rows * k).map(|_| rng.code()).collect();
    let scales: Vec<u8> = (0..rows * k / 32).map(|_| rng.scale()).collect();
    let values = (0..rows)
        .map(|r| (0..k).map(|i| decode(codes[r * k + i], scales[r * (k / 32) + i / 32])).collect())
        .collect();
    let mut bytes = codes;
    bytes.extend_from_slice(&scales);
    let code_words = rows * k / ROW_BYTES;
    let mut reads = sequential_reads(code_words, 0);
    reads.extend(sequential_reads((rows * k / 32).div_ceil(ROW_BYTES), code_words));
    Operand { bytes, reads, values }
}

fn window_operand(image: &[u8], rows: usize, full_k: usize, start_k: usize, k: usize) -> Operand {
    let scales = rows * full_k;
    let values = (0..rows)
        .map(|r| {
            (0..k)
                .map(|i| {
                    let col = start_k + i;
                    decode(image[r * full_k + col], image[scales + r * (full_k / 32) + col / 32])
                })
                .collect()
        })
        .collect();
    let mut reads = Vec::new();
    for r in 0..rows {
        for o in 0..k / ROW_BYTES {
            reads.push((r * (full_k / ROW_BYTES) + start_k / ROW_BYTES + o) as u32);
        }
    }
    for r in 0..rows {
        let first = scales + r * (full_k / 32) + start_k / 32;
        for o in 0..k / 32 {
            if o == 0 || (first + o) % ROW_BYTES == 0 {
                reads.push(((first + o) / ROW_BYTES) as u32);
            }
        }
    }
    Operand { bytes: image.to_vec(), reads, values }
}

fn f32_operand(rng: &mut Rng, rows: usize, k: usize, zero_rows: &[usize]) -> Operand {
    let values: Vec<Vec<f32>> = (0..rows)
        .map(|r| (0..k).map(|_| if zero_rows.contains(&r) { 0.0 } else { rng.float() }).collect())
        .collect();
    let bytes: Vec<u8> = values.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    let reads = sequential_reads(rows * k * 4 / ROW_BYTES, 0);
    Operand { bytes, reads, values }
}

fn mx_window_image(rng: &mut Rng, rows: usize, full_k: usize) -> Vec<u8> {
    let mut image: Vec<u8> = (0..rows * full_k).map(|_| rng.code()).collect();
    image.extend((0..rows * full_k / 32).map(|_| rng.scale()));
    image
}

struct Command {
    format: u32,
    rob_id: u32,
    banks: [u32; 3],
    groups: [u32; 3],
    m: usize,
    n: usize,
    k: usize,
    first: bool,
    last: bool,
    base: usize,
    full_k: usize,
    start_k: usize,
}

struct Chain {
    format: u32,
    m: usize,
    n: usize,
    bank: u32,
    base: usize,
    values: Vec<f32>,
}

fn step(bank_entries: u32, chain: &mut Option<Chain>, c: Command, a: Operand, b: Operand) -> Step {
    let bank_bytes = bank_entries as usize * ROW_BYTES;
    let mx = c.format == FORMAT_MXFP8 || c.format == FORMAT_WINDOW;
    if !(c.m == 1 || (c.m > 0 && c.m % 16 == 0)) || c.n == 0 || c.n % 16 != 0 || c.k == 0 || c.k % (if mx { 32 } else { 4 }) != 0 {
        panic!("mxmm: invalid shape");
    }
    if c.format == FORMAT_WINDOW && (c.full_k == 0 || c.full_k % 32 != 0 || c.start_k % 32 != 0 || c.start_k + c.k > c.full_k) {
        panic!("mxmm: invalid activation window");
    }
    if c.format != FORMAT_WINDOW && (c.full_k != c.k || c.start_k != 0) {
        panic!("mxmm: window fields are reserved");
    }
    if c.banks[0] == c.banks[1] || c.banks[0] == c.banks[2] || c.banks[1] == c.banks[2] {
        panic!("mxmm: banks must differ");
    }
    if a.bytes.len() > bank_bytes || b.bytes.len() > bank_bytes || c.base * ROW_BYTES + c.m * c.n * 4 > bank_bytes {
        panic!("mxmm: bank footprint exceeds depth");
    }
    // The accumulator buffer holds bank_entries / 4 lines of 16 results.
    if c.m * c.n > bank_entries as usize * 4 {
        panic!("mxmm: result exceeds the accumulator buffer");
    }
    if a.values.len() != c.m || b.values.len() != c.n || a.values.iter().chain(b.values.iter()).any(|v| v.len() != c.k) {
        panic!("mxmm: operand shape mismatch");
    }
    let mut live = if c.first {
        if chain.is_some() {
            panic!("mxmm: first while chain is live");
        }
        Chain { format: c.format, m: c.m, n: c.n, bank: c.banks[2], base: c.base, values: vec![0f32; c.m * c.n] }
    } else {
        chain.take().expect("mxmm: continuation without first")
    };
    if live.format != c.format || live.m != c.m || live.n != c.n || live.bank != c.banks[2] || live.base != c.base {
        panic!("mxmm: continuation changed destination");
    }
    for row in 0..c.m {
        for col in 0..c.n {
            let mut sum = live.values[row * c.n + col];
            for inner in 0..c.k {
                let (x, y) = (a.values[row][inner], b.values[col][inner]);
                sum = if c.format == FORMAT_F32 { (x * y) + sum } else { x.mul_add(y, sum) };
            }
            assert!(sum.is_finite(), "mxmm: accumulator overflow");
            live.values[row * c.n + col] = sum;
        }
    }
    let mut writes = Vec::new();
    if c.last {
        let bytes: Vec<u8> = live.values.iter().flat_map(|x| x.to_le_bytes()).collect();
        for (w, chunk) in bytes.chunks(ROW_BYTES).enumerate() {
            writes.push(Write { addr: (c.base + w) as u32, mask: 0xffff, data: word(chunk) });
        }
    } else {
        *chain = Some(live);
    }
    let mut preload = Vec::new();
    for (operand, image) in [&a.bytes, &b.bytes].into_iter().enumerate() {
        let mut padded = image.clone();
        padded.resize(image.len().div_ceil(ROW_BYTES) * ROW_BYTES, 0);
        for (row, chunk) in padded.chunks(ROW_BYTES).enumerate() {
            preload.push(Preload { bank: c.banks[operand], group: c.groups[operand], addr: row as u32, data: word(chunk) });
        }
    }
    let mut rs2 = (c.m as u64) | ((c.n as u64) << 12) | (u64::from(c.first) << 24) | (u64::from(c.last) << 25) | ((c.base as u64) << 26);
    if c.format == FORMAT_WINDOW {
        rs2 |= ((c.full_k as u64) << 32) | ((c.start_k as u64) << 48);
    }
    Step {
        cmd: MxmmCmd {
            format: c.format,
            rob_id: c.rob_id,
            iter: c.k as u32,
            rs2_lo: rs2 as u32,
            rs2_hi: (rs2 >> 32) as u32,
            op1_bank: c.banks[0],
            op2_bank: c.banks[1],
            wr_bank: c.banks[2],
            op1_group: c.groups[0],
            op2_group: c.groups[1],
            wr_group: c.groups[2],
            op1_col: 1,
            op2_col: 1,
            wr_col: 1,
        },
        preload,
        reads: [a.reads, b.reads],
        writes,
    }
}

fn command(format: u32, rob_id: u32, banks: [u32; 3], groups: [u32; 3], m: usize, n: usize, k: usize, first: bool, last: bool, base: usize) -> Command {
    Command { format, rob_id, banks, groups, m, n, k, first, last, base, full_k: k, start_k: 0 }
}

pub fn gen_case(index: u32, bank_entries: u32) -> Case {
    let mut rng = Rng(0x1234_5679 ^ index.wrapping_mul(0x9e37_79b9));
    let mut chain = None;
    let mut steps = Vec::new();
    match index {
        // One MXFP8 tile.
        0 => {
            let (a, b) = (mx_operand(&mut rng, 16, 32), mx_operand(&mut rng, 16, 32));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_MXFP8, 1, [1, 2, 3], [0, 0, 0], 16, 16, 32, true, true, 0), a, b));
        }
        // MXFP8 vector (M = 1) over two scale blocks.
        1 => {
            let (a, b) = (mx_operand(&mut rng, 1, 64), mx_operand(&mut rng, 32, 64));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_MXFP8, 2, [4, 5, 6], [1, 2, 3], 1, 32, 64, true, true, 3), a, b));
        }
        // FMA32 with two column groups.
        2 => {
            let (a, b) = (f32_operand(&mut rng, 16, 8, &[]), f32_operand(&mut rng, 32, 8, &[]));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_FMA32, 3, [7, 8, 9], [4, 5, 6], 16, 32, 8, true, true, 1), a, b));
        }
        // F32 (rounded product, rounded add) with two row groups and zero rows.
        3 => {
            let (a, b) = (f32_operand(&mut rng, 32, 4, &[0, 17]), f32_operand(&mut rng, 16, 4, &[]));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_F32, 4, [10, 11, 12], [7, 8, 9], 32, 16, 4, true, true, 2), a, b));
        }
        // MXFP8 chain: first, continuation, last.
        4 => {
            for (i, (first, last)) in [(true, false), (false, false), (false, true)].into_iter().enumerate() {
                let (a, b) = (mx_operand(&mut rng, 16, 32), mx_operand(&mut rng, 16, 32));
                steps.push(step(bank_entries, &mut chain, command(FORMAT_MXFP8, 5 + i as u32, [13, 14, 15], [10, 11, 12], 16, 16, 32, first, last, 5), a, b));
            }
        }
        // Sliding activation window over fullK = 96 accumulated as one chain.
        5 => {
            let image = mx_window_image(&mut rng, 16, 96);
            for (i, start_k) in [0usize, 32, 64].into_iter().enumerate() {
                let a = window_operand(&image, 16, 96, start_k, 32);
                let b = mx_operand(&mut rng, 16, 32);
                let c = Command { full_k: 96, start_k, ..command(FORMAT_WINDOW, 8 + i as u32, [16, 17, 18], [13, 14, 15], 16, 16, 32, i == 0, i == 2, 7) };
                steps.push(step(bank_entries, &mut chain, c, a, b));
            }
            // A single-command vector window over the upper half of fullK = 64.
            let image = mx_window_image(&mut rng, 1, 64);
            let a = window_operand(&image, 1, 64, 32, 32);
            let b = mx_operand(&mut rng, 32, 32);
            let c = Command { full_k: 64, start_k: 32, ..command(FORMAT_WINDOW, 15, [16, 17, 18], [13, 14, 15], 1, 32, 32, true, true, 4) };
            steps.push(step(bank_entries, &mut chain, c, a, b));
        }
        // Eight MXFP8 panels (two batches of contexts) filling the C bank.
        6 => {
            let (a, b) = (mx_operand(&mut rng, 32, 32), mx_operand(&mut rng, 64, 32));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_MXFP8, 11, [1021, 1022, 1023], [31, 30, 29], 32, 64, 32, true, true, 0), a, b));
        }
        // F32 chain, then an FMA32 vector after the chain closes.
        7 => {
            for (i, (first, last)) in [(true, false), (false, true)].into_iter().enumerate() {
                let (a, b) = (f32_operand(&mut rng, 16, 8, &[3]), f32_operand(&mut rng, 16, 8, &[]));
                steps.push(step(bank_entries, &mut chain, command(FORMAT_F32, 12 + i as u32, [19, 20, 21], [16, 17, 18], 16, 16, 8, first, last, 9), a, b));
            }
            let (a, b) = (f32_operand(&mut rng, 1, 12, &[]), f32_operand(&mut rng, 48, 12, &[]));
            steps.push(step(bank_entries, &mut chain, command(FORMAT_FMA32, 14, [22, 23, 24], [19, 20, 21], 1, 48, 12, true, true, 0), a, b));
        }
        _ => panic!("mxmm: unsupported case {index}"),
    }
    if chain.is_some() {
        panic!("mxmm: case {index} leaves a live chain");
    }
    Case { steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_build() {
        for index in 0..NUM_CASES {
            gen_case(index, 512);
        }
    }

    #[test]
    fn decode_edges() {
        assert_eq!(decode(0x38, 127), 1.0);
        assert_eq!(decode(0x7e, 127), 448.0);
        assert_eq!(decode(0x01, 127), 1.0 / 512.0);
        assert_eq!(decode(0x80, 127).to_bits(), 0x8000_0000);
        assert_eq!(decode(0x01, 0), f32::from_bits(1 << 22) / 512.0);
    }
}
