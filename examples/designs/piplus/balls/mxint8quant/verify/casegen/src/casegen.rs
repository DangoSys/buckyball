// MXINT8QUANT reference model, the contract shared with the BEMU model
// (emu/src/quant.rs) and Mxint8quantBall:
//
// iter = rows; rs2 = k[15:0] | in_base[29:16] | out_base[43:30] |
// columns[44] | in_pitch[58:45], rs2[63:59] zero.
// op1 bank: row i is the k/4 bank rows from in_base + i * in_pitch. With
//           columns (rows a multiple of 4) it is a row-major FP32 matrix
//           [k][rows], its row r the rows/4 bank rows from
//           in_base + r * in_pitch, and output row c is its column c.
// wr bank:  packed row i at out_base + i * stride, stride = k/16 + ceil(k/512)
//           bank rows: k INT8 codes, then the k/32 E8M0 exponent bytes, zero
//           padded to whole bank rows.
// Block exponent e of 32 values: 0 for an all-zero block, else
// max(exponent field of max |x| - 133, -127) (-127 for a subnormal maximum),
// stored as e + 127. Code: clamp(round_half_even(x / 2^e), -127, 127).
// Inputs are finite; k is a positive multiple of 32.

pub const ROW_BYTES: usize = 16;
pub const BLOCK: usize = 32;
pub const NUM_CASES: u32 = 11;

// Semantic events of a step, counted by the reference and covered by the
// testbench.
pub const ZERO_BLOCK: usize = 0;
pub const SUBNORMAL_BLOCK: usize = 1;
pub const CLAMPED_EXPONENT: usize = 2;
pub const TIE_DOWN: usize = 3;
pub const TIE_UP: usize = 4;
pub const SATURATE: usize = 5;
pub const FLUSH: usize = 6;
pub const NEGATIVE_ZERO: usize = 7;
pub const SUBNORMAL_ELEMENT: usize = 8;
pub const MAX_FINITE: usize = 9;
pub const EVENTS: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mxint8quantCmd {
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
pub struct Row {
    pub addr: u32,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Write {
    pub addr: u32,
    // 0 for a code row, else the valid exponent bytes of an exponent row.
    pub kind: u32,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub cmd: Mxint8quantCmd,
    // The input window in the op1 bank; every row is read exactly once.
    pub preload: Vec<Row>,
    pub writes: Vec<Write>,
    pub events: [u32; EVENTS],
    // Cycles the testbench holds cmdResp.ready low once the response is valid.
    pub hold: u32,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub steps: Vec<Step>,
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub rows: usize,
    pub k: usize,
    pub columns: bool,
    pub in_base: usize,
    pub out_base: usize,
    pub pitch: usize,
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

/// Bank rows of one packed row of length k.
fn packed_rows(k: usize) -> usize {
    k / 16 + k.div_ceil(512)
}

/// The block exponent e (stored as e + 127) of 32 FP32 values.
fn block_exponent(values: &[u32], events: &mut [u32; EVENTS]) -> i32 {
    let maximum = values.iter().map(|bits| bits & 0x7fff_ffff).max().unwrap();
    if maximum == 0 {
        events[ZERO_BLOCK] += 1;
        return 0;
    }
    let biased = (maximum >> 23) as i32;
    if biased == 0 {
        events[SUBNORMAL_BLOCK] += 1;
        return -127;
    }
    if biased - 133 < -127 {
        events[CLAMPED_EXPONENT] += 1;
    }
    if biased == 254 {
        events[MAX_FINITE] += 1;
    }
    (biased - 133).max(-127)
}

/// clamp(round_half_even(x / 2^e), -127, 127) of FP32 bits x.
fn encode(bits: u32, exponent: i32, events: &mut [u32; EVENTS]) -> u8 {
    let field = ((bits >> 23) & 0xff) as i32;
    let significand = (bits & 0x7f_ffff) | if field == 0 { 0 } else { 0x80_0000 };
    let shift = 150 + exponent - field.max(1);
    assert!(shift >= 17, "mxint8quant: element exceeds its block maximum");
    if bits == 0x8000_0000 {
        events[NEGATIVE_ZERO] += 1;
    }
    if field == 0 && significand != 0 && exponent != -127 {
        events[SUBNORMAL_ELEMENT] += 1;
    }
    let rounded = if shift > 31 {
        0
    } else {
        let quotient = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if remainder == half {
            events[if quotient & 1 == 1 { TIE_UP } else { TIE_DOWN }] += 1;
        }
        quotient + u32::from(remainder > half || (remainder == half && quotient & 1 == 1))
    };
    if rounded > 127 {
        events[SATURATE] += 1;
    }
    if rounded == 0 && significand != 0 {
        events[FLUSH] += 1;
    }
    let magnitude = rounded.min(127) as i32;
    (if bits >> 31 == 1 { -magnitude } else { magnitude }) as i8 as u8
}

/// The packed output rows of `values` (rows x k, output row after row).
fn quantize(shape: &Shape, values: &[u32], events: &mut [u32; EVENTS]) -> Vec<u8> {
    let stride = packed_rows(shape.k);
    let mut packed = vec![0u8; shape.rows * stride * ROW_BYTES];
    for (row, values) in values.chunks_exact(shape.k).enumerate() {
        let packed = &mut packed[row * stride * ROW_BYTES..(row + 1) * stride * ROW_BYTES];
        for (block, values) in values.chunks_exact(BLOCK).enumerate() {
            let exponent = block_exponent(values, events);
            packed[shape.k + block] = (exponent + 127) as u8;
            for (index, bits) in values.iter().enumerate() {
                packed[block * BLOCK + index] = encode(*bits, exponent, events);
            }
        }
    }
    packed
}

fn word(bytes: &[u8]) -> u128 {
    u128::from_le_bytes(bytes.try_into().unwrap())
}

fn step(
    entries: usize,
    rob_id: u32,
    banks: [u32; 2],
    groups: [u32; 2],
    shape: Shape,
    values: &[u32],
    hold: u32,
) -> Step {
    let Shape {
        rows,
        k,
        columns,
        in_base,
        out_base,
        pitch,
    } = shape;
    assert!(
        rows > 0 && k > 0 && k % BLOCK == 0 && (!columns || rows % 4 == 0),
        "mxint8quant: rows > 0 (a multiple of 4 with columns) and K a positive multiple of 32"
    );
    assert!(k < 1 << 16 && in_base < 1 << 14 && out_base < 1 << 14 && pitch < 1 << 14);
    assert!(banks[0] != banks[1], "mxint8quant: input and output banks must differ");
    assert_eq!(values.len(), rows * k);
    assert!(
        values.iter().all(|bits| bits & 0x7fff_ffff < 0x7f80_0000),
        "mxint8quant: non-finite input"
    );
    // The bank rows of an input row and the input rows.
    let (width, count) = if columns { (rows / 4, k) } else { (k / 4, rows) };
    assert!(pitch >= width, "mxint8quant: input rows overlap");
    let stride = packed_rows(k);
    assert!(
        in_base + (count - 1) * pitch + width <= entries && out_base + rows * stride <= entries,
        "mxint8quant: bank window exceeds capacity"
    );
    // Scatter the values to the input bank: with columns, element r of output
    // row c is element c of input row r.
    let mut image = std::collections::BTreeMap::<usize, [u8; ROW_BYTES]>::new();
    for (i, bits) in values.iter().enumerate() {
        let (input_row, element) = if columns { (i % k, i / k) } else { (i / k, i % k) };
        let row = image
            .entry(in_base + input_row * pitch + element / 4)
            .or_insert([0; ROW_BYTES]);
        row[(element % 4) * 4..(element % 4) * 4 + 4].copy_from_slice(&bits.to_le_bytes());
    }
    assert_eq!(image.len(), count * width);
    let preload = image
        .iter()
        .map(|(addr, bytes)| Row {
            addr: *addr as u32,
            data: word(bytes),
        })
        .collect();
    let mut events = [0; EVENTS];
    let packed = quantize(&shape, values, &mut events);
    let blocks = k / BLOCK;
    let writes = packed
        .chunks_exact(ROW_BYTES)
        .enumerate()
        .map(|(i, bytes)| {
            let within = i % stride;
            let kind = if within < k / 16 {
                0
            } else {
                (blocks - (within - k / 16) * ROW_BYTES).min(ROW_BYTES)
            };
            Write {
                addr: (out_base + i) as u32,
                kind: kind as u32,
                data: word(bytes),
            }
        })
        .collect();
    let rs2 =
        k as u64 | (in_base as u64) << 16 | (out_base as u64) << 30 | u64::from(columns) << 44 | (pitch as u64) << 45;
    Step {
        cmd: Mxint8quantCmd {
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
        writes,
        events,
        hold,
    }
}

/// 32 random values whose largest exponent field is `top`, the others up to
/// `spread` below it; with rounding ties, signed zeros and saturating maxima.
fn random_block(rng: &mut Rng, top: u32, spread: u32) -> Vec<u32> {
    let exponent = if top == 0 { -127 } else { (top as i32 - 133).max(-127) };
    (0..BLOCK)
        .map(|i| {
            let sign = (rng.next() & 1) << 31;
            let field = if i == 0 {
                top
            } else {
                top.saturating_sub(rng.next() % spread)
            };
            let mut mantissa = rng.next() & 0x7f_ffff;
            match rng.next() % 16 {
                0 if i != 0 => return sign,
                1 | 2 => {
                    let shift = 150 + exponent - field.max(1) as i32;
                    if (1..=23).contains(&shift) {
                        mantissa = (mantissa & !((1 << shift) - 1)) | 1 << (shift - 1);
                    }
                }
                3 => mantissa |= 0x7f_0000,
                _ => {}
            }
            if i == 0 {
                mantissa |= 1;
            }
            sign | field << 23 | mantissa
        })
        .collect()
}

/// Random blocks over the exponent range: any, around one, tiny and large.
fn random_values(rng: &mut Rng, count: usize) -> Vec<u32> {
    (0..count / BLOCK)
        .flat_map(|_| {
            let top = match rng.next() % 4 {
                0 => rng.next() % 255,
                1 => 120 + rng.next() % 20,
                2 => rng.next() % 6,
                _ => 200 + rng.next() % 55,
            };
            let spread = 1 + rng.next() % 30;
            random_block(rng, top, spread)
        })
        .collect()
}

fn random_shape(rng: &mut Rng, entries: usize) -> Shape {
    loop {
        let columns = rng.next() % 2 == 1;
        let (rows, k) = if columns {
            (
                4 * (1 + rng.next() as usize % 8),
                BLOCK * (1 + rng.next() as usize % 12),
            )
        } else {
            (1 + rng.next() as usize % 5, BLOCK * (1 + rng.next() as usize % 40))
        };
        let (width, count) = if columns { (rows / 4, k) } else { (k / 4, rows) };
        let pitch = width + rng.next() as usize % 3;
        let input = (count - 1) * pitch + width;
        let output = rows * packed_rows(k);
        if input > entries || output > entries {
            continue;
        }
        let in_base = rng.next() as usize % (entries - input + 1);
        let out_base = rng.next() as usize % (entries - output + 1);
        return Shape {
            rows,
            k,
            columns,
            in_base,
            out_base,
            pitch,
        };
    }
}

fn random_step(rng: &mut Rng, entries: usize, hold: u32) -> Step {
    let shape = random_shape(rng, entries);
    let values = random_values(rng, shape.rows * shape.k);
    let input = rng.next() % 1024;
    let output = (input + 1 + rng.next() % 1023) % 1024;
    let groups = [rng.next() % 24, rng.next() % 24];
    step(entries, rng.next() % 8, [input, output], groups, shape, &values, hold)
}

fn f(value: f32) -> u32 {
    value.to_bits()
}

pub fn gen_case(index: u32, bank_entries: u32) -> Case {
    let entries = bank_entries as usize;
    let mut rng = Rng(0x2545_f491 ^ index.wrapping_mul(0x9e37_79b9));
    let shape = |rows, k, columns, in_base, out_base, pitch| Shape {
        rows,
        k,
        columns,
        in_base,
        out_base,
        pitch,
    };
    let steps = match index {
        // quant_edges_test: two rows of two blocks, the rows two bank rows
        // apart; ties to even, the clamp at 127, signed and all-zero blocks.
        0 => {
            let mut values = vec![0u32; 2 * 64];
            let row0 = [127.5, -127.5, 0.5, 1.5, 2.5, -2.5, 3.5, -0.5, -0.0, 126.49999];
            let row1 = [254.0, 3.0, 5.0, -7.0];
            let tail = [1.0, 0.5, 0.25, 1e-3];
            for (i, v) in row0.iter().enumerate() {
                values[i] = f(*v);
            }
            for i in 0..6 {
                values[64 + i] = f(row1[i % 4]);
            }
            for (i, v) in tail.iter().enumerate() {
                values[64 + 32 + i] = f(*v);
            }
            let step = step(entries, 1, [1, 2], [0, 0], shape(2, 64, false, 3, 2, 18), &values, 0);
            // The golden bytes of quant_edges_test pin the reference.
            let byte = |i: usize| (step.writes[i / 16].data >> (i % 16 * 8)) as u8;
            let codes: Vec<i8> = (0..10).map(|i| byte(i) as i8).collect();
            assert_eq!(codes, [127, -127, 0, 2, 2, -2, 4, 0, 0, 126]);
            let codes: Vec<i8> = [80, 81, 82, 83, 84, 85, 112, 113, 114, 115]
                .iter()
                .map(|i| byte(*i) as i8)
                .collect();
            assert_eq!(codes, [127, 2, 2, -4, 127, 2, 64, 32, 16, 0]);
            assert_eq!([byte(64), byte(65), byte(80 + 64), byte(80 + 65)], [127, 127, 128, 121]);
            vec![step]
        }
        // Special blocks: signed zeros, the largest finite magnitudes, a
        // subnormal maximum, a clamped exponent, ties, saturation and flushed
        // values; a second row of random blocks.
        1 => {
            let mut values = Vec::new();
            values.extend((0..BLOCK).map(|i| if i % 2 == 1 { 0x8000_0000 } else { 0 }));
            values.extend((0..BLOCK).map(|i| match i {
                0 => 0x7f7f_ffff,
                1 => 0xff7f_ffff,
                _ => (rng.next() & 0x807f_ffff) | (220 + rng.next() % 34) << 23,
            }));
            let subnormal = [
                1,
                0x7f_ffff,
                0x8000_0001,
                0x40_0000,
                0x20_0000,
                0x60_0000,
                0x8060_0000,
                0x2f_ffff,
            ];
            values.extend((0..BLOCK).map(|i| subnormal[i % 8]));
            values.extend((0..BLOCK).map(|i| (rng.next() & 0x807f_ffff) | (1 + i as u32 % 3) << 23));
            values.extend((0..BLOCK).map(|i| match i {
                0 => f(127.5),
                _ => f(if i % 2 == 1 { -1.0 } else { 1.0 } * (i as f32 - 0.5)),
            }));
            let saturate = [127.75, -127.5, 126.5, 127.4, -126.5, 125.5, -0.5, 0.75];
            values.extend((0..BLOCK).map(|i| f(saturate[i % 8])));
            let flush = [1048576.0, 1.0, 0.0009765625, 8192.0, 24576.0, -8192.0, 1e-40, 16384.0];
            values.extend((0..BLOCK).map(|i| f(flush[i % 8])));
            values.extend(random_block(&mut rng, 200, 30));
            values.extend(random_values(&mut rng, 256));
            vec![step(
                entries,
                6,
                [5, 9],
                [3, 17],
                shape(2, 256, false, 100, 900, 70),
                &values,
                0,
            )]
        }
        // Rows of 17 blocks: a second exponent row per output row.
        2 => {
            let values = random_values(&mut rng, 3 * 544);
            vec![step(
                entries,
                2,
                [12, 7],
                [23, 1],
                shape(3, 544, false, 5, 7, 137),
                &values,
                0,
            )]
        }
        // Rows of exactly 16 blocks: one full exponent row.
        3 => {
            let values = random_values(&mut rng, 2 * 512);
            vec![step(
                entries,
                3,
                [0, 1],
                [2, 0],
                shape(2, 512, false, 0, 300, 128),
                &values,
                0,
            )]
        }
        // One row of 4096 values: the input fills the bank, the output ends
        // at its last row.
        4 => {
            let values: Vec<u32> = (0..128)
                .flat_map(|_| {
                    let top = 118 + rng.next() % 30;
                    random_block(&mut rng, top, 12)
                })
                .collect();
            vec![step(
                entries,
                7,
                [1023, 512],
                [7, 8],
                shape(1, 4096, false, 0, entries - 264, 1024),
                &values,
                0,
            )]
        }
        // Columns: the transpose of a 32 x 4 matrix.
        5 => {
            let values = random_values(&mut rng, 4 * 32);
            vec![step(
                entries,
                0,
                [3, 4],
                [0, 0],
                shape(4, 32, true, 0, 0, 1),
                &values,
                0,
            )]
        }
        // Columns 8..15 of a [64][16] matrix whose rows are four bank rows.
        6 => {
            let values = random_values(&mut rng, 8 * 64);
            vec![step(
                entries,
                5,
                [20, 21],
                [4, 5],
                shape(8, 64, true, 12, 40, 4),
                &values,
                0,
            )]
        }
        // Columns of 17 blocks: a second exponent row per column.
        7 => {
            let values = random_values(&mut rng, 4 * 544);
            vec![step(
                entries,
                4,
                [8, 30],
                [9, 10],
                shape(4, 544, true, 300, 0, 1),
                &values,
                0,
            )]
        }
        // Columns: the transpose of a 256 x 16 matrix that fills the bank.
        8 => {
            let values = random_values(&mut rng, 16 * 256);
            vec![step(
                entries,
                1,
                [600, 601],
                [11, 12],
                shape(16, 256, true, 0, 600, 4),
                &values,
                0,
            )]
        }
        // Random commands back to back without reset.
        9 => (0..12).map(|_| random_step(&mut rng, entries, 0)).collect(),
        // A response held for 40 cycles, then a command right after it.
        10 => vec![random_step(&mut rng, entries, 40), random_step(&mut rng, entries, 0)],
        _ => panic!("mxint8quant: unsupported case {index}"),
    };
    Case { steps }
}
