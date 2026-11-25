// MXINT8QUANT reference model, the contract shared with the BEMU model
// (emu/src/quant.rs) and Mxint8quantizeBall:
//
// rs1:      iter = rows, op1 = input bank, wr = output bank (both one group).
// rs2:      k in [15:0], in_base in [29:16], out_base in [43:30], rest zero.
// op1 bank: row r = k little-endian FP32 at rows in_base + r*k/4 .., read in
//           order, eight bank rows per block of 32 values.
// wr bank:  row r at out_base + r*S(k), S(k) = k/16 + ceil(k/512): k INT8
//           codes, then the exponent bytes e + 127, zero-padded. Per block two
//           code rows, and at every 16th or the row's last block its
//           exponent row, all with full masks.
// The block exponent is max(F - 133, -127) of the largest magnitude
// (0 for an all-zero block); codes round half to even and clamp to +-127.

pub const ROW_BYTES: usize = 16;
pub const BLOCK: usize = 32;
pub const NUM_CASES: u32 = 9;
pub const NUM_EVENTS: u32 = 11;

// Numeric events counted per step, sampled by the coverage collector.
pub const ZERO_BLOCK: usize = 0; // every magnitude zero: exponent byte 127
pub const SUBNORMAL_MAX: usize = 1; // subnormal block maximum: e = -127
pub const CLAMPED_EXPONENT: usize = 2; // F in 1..=6: F - 133 clamps to -127
pub const TOP_EXPONENT: usize = 3; // F = 254: exponent byte 248
pub const TIE_DOWN: usize = 4; // exact half step, even quotient kept
pub const TIE_UP: usize = 5; // exact half step, odd quotient rounded up
pub const SATURATE: usize = 6; // rounded magnitude 128 clamped to 127
pub const FLUSH: usize = 7; // nonzero value with shift > 31: code 0
pub const NEGATIVE_ZERO: usize = 8; // -0.0 input: code 0
pub const SUBNORMAL_INPUT: usize = 9; // subnormal non-maximum input
pub const NEGATIVE_CODE: usize = 10; // nonzero negative code

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mxint8quantizeCmd {
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
    pub cmd: Mxint8quantizeCmd,
    pub preload: Vec<Preload>,
    pub reads: Vec<u32>,
    pub writes: Vec<Write>,
    pub events: [u32; NUM_EVENTS as usize],
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
}

fn word(bytes: &[u8]) -> u128 {
    let mut out = 0u128;
    for (lane, byte) in bytes.iter().enumerate() {
        out |= u128::from(*byte) << (lane * 8);
    }
    out
}

/// Bank rows of one packed row of length k.
fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

/// The block exponent e (stored as e + 127) of 32 FP32 values.
fn block_exponent(values: &[u32], events: &mut [u32]) -> i32 {
    let maximum = values.iter().map(|bits| bits & 0x7fff_ffff).max().unwrap();
    if maximum == 0 {
        events[ZERO_BLOCK] += 1;
        return 0;
    }
    let biased = (maximum >> 23) as i32;
    if biased == 0 {
        events[SUBNORMAL_MAX] += 1;
        return -127;
    }
    if biased < 7 {
        events[CLAMPED_EXPONENT] += 1;
    }
    if biased == 254 {
        events[TOP_EXPONENT] += 1;
    }
    (biased - 133).max(-127)
}

/// clamp(round_half_even(x / 2^e), -127, 127) of FP32 bits x.
fn encode(bits: u32, exponent: i32, events: &mut [u32]) -> u8 {
    let field = ((bits >> 23) & 0xff) as i32;
    let significand = (bits & 0x7f_ffff) | if field == 0 { 0 } else { 0x80_0000 };
    let shift = 150 + exponent - field.max(1);
    assert!(
        shift >= 17,
        "mxint8quantize: element exceeds its block maximum"
    );
    if bits == 0x8000_0000 {
        events[NEGATIVE_ZERO] += 1;
    }
    let magnitude = if shift > 31 {
        if significand != 0 {
            events[FLUSH] += 1;
        }
        0
    } else {
        let quotient = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if remainder == half {
            events[if quotient & 1 == 1 { TIE_UP } else { TIE_DOWN }] += 1;
        }
        let rounded =
            quotient + u32::from(remainder > half || (remainder == half && quotient & 1 == 1));
        if rounded > 127 {
            events[SATURATE] += 1;
        }
        rounded.min(127)
    } as i32;
    if bits >> 31 == 1 && magnitude != 0 {
        events[NEGATIVE_CODE] += 1;
    }
    (if bits >> 31 == 1 {
        -magnitude
    } else {
        magnitude
    }) as i8 as u8
}

/// The packed rows (S(k) * 16 bytes each) of `rows` rows of k values.
fn quantize(input: &[u32], k: usize, events: &mut [u32]) -> Vec<u8> {
    let stride = packed_rows(k) * ROW_BYTES;
    let mut out = vec![0u8; input.len() / k * stride];
    for (row, values) in input.chunks_exact(k).enumerate() {
        let packed = &mut out[row * stride..(row + 1) * stride];
        for (block, values) in values.chunks_exact(BLOCK).enumerate() {
            let maximum = values.iter().map(|bits| bits & 0x7fff_ffff).max().unwrap();
            let exponent = block_exponent(values, events);
            packed[k + block] = (exponent + 127) as u8;
            for (index, bits) in values.iter().enumerate() {
                if bits & 0x7f80_0000 == 0 && bits & 0x7f_ffff != 0 && bits & 0x7fff_ffff != maximum
                {
                    events[SUBNORMAL_INPUT] += 1;
                }
                packed[block * BLOCK + index] = encode(*bits, exponent, events);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn step(
    bank_entries: u32,
    rob_id: u32,
    banks: [u32; 2],
    groups: [u32; 2],
    k: usize,
    in_base: usize,
    out_base: usize,
    input: &[u32],
) -> Step {
    let entries = bank_entries as usize;
    let rows = input.len() / k;
    let stride = packed_rows(k);
    assert!(
        rows > 0 && k > 0 && k % BLOCK == 0 && input.len() == rows * k,
        "mxint8quantize: rows > 0 and K a positive multiple of 32"
    );
    assert!(
        in_base + rows * k / 4 <= entries && out_base + rows * stride <= entries,
        "mxint8quantize: bank window exceeds capacity"
    );
    assert!(
        banks[0] != banks[1],
        "mxint8quantize: input and output banks must differ"
    );
    assert!(
        input.iter().all(|bits| bits & 0x7fff_ffff < 0x7f80_0000),
        "mxint8quantize: non-finite input"
    );
    let mut events = [0u32; NUM_EVENTS as usize];
    let packed = quantize(input, k, &mut events);
    let mut preload = Vec::new();
    let mut reads = Vec::new();
    for (row, values) in input.chunks(4).enumerate() {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let addr = (in_base + row) as u32;
        preload.push(Preload {
            bank: banks[0],
            group: groups[0],
            addr,
            data: word(&bytes),
        });
        reads.push(addr);
    }
    let mut writes = Vec::new();
    let blocks = k / BLOCK;
    for row in 0..rows {
        let base = out_base + row * stride;
        let bank_row = |offset: usize| {
            let start = (row * stride + offset) * ROW_BYTES;
            Write {
                addr: (base + offset) as u32,
                mask: 0xffff,
                data: word(&packed[start..start + ROW_BYTES]),
            }
        };
        for block in 0..blocks {
            writes.push(bank_row(2 * block));
            writes.push(bank_row(2 * block + 1));
            if block % 16 == 15 || block == blocks - 1 {
                writes.push(bank_row(k / 16 + block / 16));
            }
        }
    }
    let rs2 = k as u64 | (in_base as u64) << 16 | (out_base as u64) << 30;
    Step {
        cmd: Mxint8quantizeCmd {
            rob_id,
            iter: rows as u32,
            rs2_lo: rs2 as u32,
            rs2_hi: (rs2 >> 32) as u32,
            op1_bank: banks[0],
            op2_bank: 0,
            wr_bank: banks[1],
            op1_group: groups[0],
            op2_group: 0,
            wr_group: groups[1],
            op1_col: 1,
            op2_col: 0,
            wr_col: 1,
        },
        preload,
        reads,
        writes,
        events,
    }
}

/// 32 random finite values: the largest biased exponent is `top` (exactly
/// once), the others lie up to `spread` below it, signs random.
fn block(rng: &mut Rng, top: u32, spread: u32) -> Vec<u32> {
    let peak = (rng.next() % BLOCK as u32) as usize;
    (0..BLOCK)
        .map(|i| {
            let field = if i == peak {
                top
            } else {
                top.saturating_sub(rng.next() % (spread + 1))
            };
            let bits = (rng.next() & 0x807f_ffff) | (field << 23);
            if i == peak && bits & 0x7fff_ffff == 0 {
                bits | 1
            } else {
                bits
            }
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

pub fn gen_case(index: u32, bank_entries: u32) -> Case {
    let entries = bank_entries as usize;
    let mut rng = Rng(0x2545_f491 ^ index.wrapping_mul(0x9e37_79b9));
    let steps = match index {
        // quant_edges_test: two rows of K = 64, in_base 3, out_base 2. Row 0
        // block 0 maximum 127.5 (ties to even, the clamp at 127, signed
        // zeros), row 0 block 1 all zero, row 1 block 0 maximum 254
        // (exponent 1), row 1 block 1 maximum 1 (exponent -6). The reference
        // must reproduce the test's hand-derived bytes.
        0 => {
            let row0 = [
                127.5f32, -127.5, 0.5, 1.5, 2.5, -2.5, 3.5, -0.5, -0.0, 126.49999,
            ];
            let row1 = [254.0f32, 3.0, 5.0, -7.0, 254.0, 3.0];
            let row1b = [1.0f32, 0.5, 0.25, 1e-3];
            let mut x = vec![0f32; 128];
            x[..10].copy_from_slice(&row0);
            x[64..70].copy_from_slice(&row1);
            x[96..100].copy_from_slice(&row1b);
            let input = bits(&x);
            let mut events = [0u32; NUM_EVENTS as usize];
            let packed = quantize(&input, 64, &mut events);
            let codes: [i8; 10] = [127, -127, 0, 2, 2, -2, 4, 0, 0, 126];
            for (i, code) in codes.iter().enumerate() {
                assert_eq!(
                    packed[i] as i8, *code,
                    "mxint8quantize: edges row 0 code {i}"
                );
            }
            let codes: [i8; 6] = [127, 2, 2, -4, 127, 2];
            for (i, code) in codes.iter().enumerate() {
                assert_eq!(
                    packed[80 + i] as i8,
                    *code,
                    "mxint8quantize: edges row 1 code {i}"
                );
            }
            let codes: [i8; 4] = [64, 32, 16, 0];
            for (i, code) in codes.iter().enumerate() {
                assert_eq!(
                    packed[80 + 32 + i] as i8,
                    *code,
                    "mxint8quantize: edges row 1 code {}",
                    32 + i
                );
            }
            assert_eq!(
                packed[64..66],
                [127, 127],
                "mxint8quantize: edges row 0 exponents"
            );
            assert_eq!(
                packed[80 + 64..80 + 66],
                [128, 121],
                "mxint8quantize: edges row 1 exponents"
            );
            vec![step(bank_entries, 1, [1, 2], [0, 0], 64, 3, 2, &input)]
        }
        // Small and extreme block maxima, one row of eight blocks: subnormal
        // maximum, maxima F = 1..6 (clamped), F = 7, signed zeros only, a
        // large maximum over tiny and subnormal values (flush), F = 254, the
        // most negative finite value, and values around one.
        1 => {
            let mut input = Vec::new();
            input.extend(block(&mut rng, 0, 0));
            let top = 1 + rng.next() % 6;
            input.extend(block(&mut rng, top, 6));
            input.extend(block(&mut rng, 7, 7));
            input.extend((0..BLOCK as u32).map(|i| if i % 3 == 0 { 0x8000_0000 } else { 0 }));
            input.extend(block(&mut rng, 200, 200));
            input.extend(block(&mut rng, 254, 40));
            let mut b = block(&mut rng, 254, 8);
            let peak = b
                .iter()
                .position(|x| x & 0x7f80_0000 == 0x7f00_0000)
                .unwrap();
            b[peak] = 0xff7f_ffff;
            input.extend(b);
            input.extend(block(&mut rng, 127, 8));
            vec![step(bank_entries, 2, [3, 4], [1, 2], 256, 0, 0, &input)]
        }
        // Exponent sweep over the full input capacity: two rows of
        // K = 2 * entries, block g with maximum F = 1 + 2g, the last F = 254.
        2 => {
            let k = 2 * entries;
            let blocks = 2 * k / BLOCK;
            let input: Vec<u32> = (0..blocks)
                .flat_map(|g| {
                    let top = if g == blocks - 1 {
                        254
                    } else {
                        (1 + 2 * g as u32).min(253)
                    };
                    block(&mut rng, top, 30)
                })
                .collect();
            vec![step(bank_entries, 5, [6, 7], [3, 4], k, 0, 0, &input)]
        }
        // The largest K (one row fills the input bank), the output ending at
        // the last bank row; every block of a random maximum and spread.
        3 => {
            let k = 4 * entries;
            let input: Vec<u32> = (0..k / BLOCK)
                .flat_map(|_| {
                    let top = rng.next() % 255;
                    let spread = 1 + rng.next() % 40;
                    block(&mut rng, top, spread)
                })
                .collect();
            vec![step(
                bank_entries,
                6,
                [8, 9],
                [5, 6],
                k,
                0,
                entries - packed_rows(k),
                &input,
            )]
        }
        // The most rows of one block: the input fills the bank, the output
        // ends at the last bank row.
        4 => {
            let rows = entries / 8;
            let input: Vec<u32> = (0..rows)
                .flat_map(|_| {
                    let top = 100 + rng.next() % 60;
                    block(&mut rng, top, 24)
                })
                .collect();
            vec![step(
                bank_entries,
                3,
                [10, 11],
                [7, 8],
                BLOCK,
                entries - rows * 8,
                entries - rows * 3,
                &input,
            )]
        }
        // quant_random_test shape: three rows of K = 544 (17 blocks, a
        // second exponent row holding one byte), a different scale per block.
        5 => {
            let input: Vec<u32> = (0..3 * 17)
                .flat_map(|g| block(&mut rng, 97 + (g as u32 * 7) % 61, 24))
                .collect();
            vec![step(bank_entries, 4, [12, 13], [9, 10], 544, 5, 7, &input)]
        }
        // Exact half steps and saturation: two rows of K = 512 (one exponent
        // row, written at the row's 16th and last block). Block maximum
        // +-127.5 * 2^e; the others (n + 0.5) * 2^e, n * 2^e or 127.75 * 2^e.
        6 => {
            let input: Vec<u32> = (0..32)
                .flat_map(|_| {
                    let e = (rng.next() % 41) as i32 - 20;
                    let scale = 2f32.powi(e);
                    let peak = (rng.next() % BLOCK as u32) as usize;
                    let values: Vec<f32> = (0..BLOCK)
                        .map(|i| {
                            let sign = if rng.next() & 1 == 1 { -1.0 } else { 1.0 };
                            let n = (rng.next() % 127) as f32;
                            let magnitude = match (i == peak, rng.next() % 4) {
                                (true, _) => 127.5,
                                (_, 0) => n,
                                (_, 3) => 127.75,
                                _ => n + 0.5,
                            };
                            sign * magnitude * scale
                        })
                        .collect();
                    bits(&values)
                })
                .collect();
            vec![step(bank_entries, 7, [14, 15], [11, 12], 512, 0, 0, &input)]
        }
        // Back-to-back commands without reset: the widest bank ids, the
        // highest groups, every ROB id parity, swapped bank roles.
        7 => {
            let shapes: [(u32, [u32; 2], [u32; 2], usize, usize, usize, usize); 4] = [
                (0, [0, 1023], [0, 23], 1, 32, 0, 0),
                (7, [1023, 0], [23, 0], 3, 96, 100, 600),
                (3, [517, 4], [7, 9], 2, 1024, 300, 11),
                (5, [12, 13], [11, 11], 5, 160, 17, 900),
            ];
            shapes
                .iter()
                .map(|&(rob_id, banks, groups, rows, k, in_base, out_base)| {
                    let input: Vec<u32> = (0..rows * k / BLOCK)
                        .flat_map(|_| {
                            let top = 60 + rng.next() % 140;
                            block(&mut rng, top, 32)
                        })
                        .collect();
                    step(
                        bank_entries,
                        rob_id,
                        banks,
                        groups,
                        k,
                        in_base,
                        out_base,
                        &input,
                    )
                })
                .collect()
        }
        // Random commands back to back: K, rows, bases, banks, groups, ROB
        // ids and per-block scales; zeros of both signs sprinkled in.
        8 => (0..16)
            .map(|_| {
                let k = BLOCK * (1 + (rng.next() % 32) as usize);
                let limit = (entries / (k / 4)).min(entries / packed_rows(k)).min(8);
                let rows = 1 + (rng.next() as usize) % limit;
                let in_base = (rng.next() as usize) % (entries - rows * k / 4 + 1);
                let out_base = (rng.next() as usize) % (entries - rows * packed_rows(k) + 1);
                let input_bank = rng.next() % 1024;
                let output_bank = (input_bank + 1 + rng.next() % 1023) % 1024;
                let groups = [rng.next() % 24, rng.next() % 24];
                let rob_id = rng.next() % 8;
                let input: Vec<u32> = (0..rows * k / BLOCK)
                    .flat_map(|_| {
                        let top = rng.next() % 255;
                        let spread = rng.next() % 64;
                        let mut values = block(&mut rng, top, spread);
                        for value in values.iter_mut() {
                            if rng.next() % 16 == 0 && (*value >> 23) & 0xff != top {
                                *value &= 0x8000_0000;
                            }
                        }
                        values
                    })
                    .collect();
                step(
                    bank_entries,
                    rob_id,
                    [input_bank, output_bank],
                    groups,
                    k,
                    in_base,
                    out_base,
                    &input,
                )
            })
            .collect(),
        _ => panic!("mxint8quantize: unsupported case {index}"),
    };
    Case { steps }
}
