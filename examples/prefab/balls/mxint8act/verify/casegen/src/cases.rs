//! Directed and random MXINT8ACT cases: bank images before and after.
use crate::reference::{activate, row_stride, EVENTS};

pub const DIRECTED: u32 = 15;
pub const NUM_CASES: u32 = DIRECTED + 24;

pub struct Case {
    pub rows: usize,
    pub k: usize,
    pub gated: bool,
    pub in_base: usize,
    pub out_base: usize,
    pub table_base: usize,
    /// Cycles the response stays back-pressured.
    pub hold: u32,
    /// Input, table, output before and expected output, one bank each.
    pub banks: [Vec<u8>; 4],
    pub events: [u32; EVENTS],
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn code(&mut self) -> i8 {
        (self.below(255) as i32 - 127) as i8
    }
}

const EDGE_EXPONENTS: [i32; 9] = [-127, -25, -24, -11, -10, -9, 4, 5, 127];

fn random_exponent(rng: &mut Rng) -> i32 {
    match rng.below(20) {
        0..=13 => rng.below(39) as i32 - 30,
        14..=16 => rng.below(255) as i32 - 127,
        _ => EDGE_EXPONENTS[rng.below(EDGE_EXPONENTS.len() as u64) as usize],
    }
}

fn random_codes(rng: &mut Rng) -> [i8; 32] {
    let kind = rng.below(10);
    let mut codes = [0i8; 32];
    for code in codes.iter_mut() {
        *code = match kind {
            0 => 0,
            1 => (rng.below(7) as i32 - 3) as i8,
            2 => [-127, -1, 0, 1, 127][rng.below(5) as usize],
            _ => rng.code(),
        };
    }
    codes
}

fn random_block(rng: &mut Rng) -> (i32, [i8; 32]) {
    (random_exponent(rng), random_codes(rng))
}

fn gelu_table() -> Vec<i32> {
    (0..4096)
        .map(|i| {
            let x = i as f64 / 128.0 - 16.0;
            let y = 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            (y * (1u64 << 26) as f64).round_ties_even() as i32
        })
        .collect()
}

fn random_table(rng: &mut Rng) -> Vec<i32> {
    (0..4096).map(|_| rng.next() as i32).collect()
}

/// One case: `block(rng, row, block, gate)` gives each block's exponent and codes.
#[allow(clippy::too_many_arguments)]
fn build(
    rng: &mut Rng,
    entries: usize,
    rows: usize,
    k: usize,
    gated: bool,
    in_base: usize,
    out_base: usize,
    table_base: usize,
    table: &[i32],
    mut block: impl FnMut(&mut Rng, usize, usize, bool) -> (i32, [i8; 32]),
) -> Case {
    let stride = row_stride(k) * 16;
    let bytes = entries * 16;
    // Padding and everything outside the spans hold garbage the Ball must ignore.
    let garbage = |rng: &mut Rng| (0..bytes).map(|_| rng.next() as u8).collect::<Vec<u8>>();
    let mut input = garbage(rng);
    let mut table_bank = garbage(rng);
    let before = garbage(rng);
    for part in 0..1 + usize::from(gated) {
        for row in 0..rows {
            let source = in_base * 16 + (part * rows + row) * stride;
            for b in 0..k / 32 {
                let (exponent, codes) = block(rng, row, b, part == 1);
                assert!((-127..=127).contains(&exponent));
                input[source + k + b] = (exponent + 127) as u8;
                for (lane, code) in codes.iter().enumerate() {
                    assert!(*code != -128);
                    input[source + b * 32 + lane] = *code as u8;
                }
            }
        }
    }
    for (i, value) in table.iter().enumerate() {
        table_bank[table_base * 16 + i * 4..table_base * 16 + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    let mut expected = before.clone();
    let mut events = [0u32; EVENTS];
    activate(
        &input,
        &table_bank,
        &mut expected,
        rows,
        k,
        gated,
        in_base,
        out_base,
        table_base,
        &mut events,
    );
    Case {
        rows,
        k,
        gated,
        in_base,
        out_base,
        table_base,
        hold: 0,
        banks: [input, table_bank, before, expected],
        events,
    }
}

pub fn gen_case(index: u32, entries: usize) -> Case {
    assert!(index < NUM_CASES && entries >= 1024);
    let rng = &mut Rng(0x9e37_79b9_7f4a_7c15 ^ (u64::from(index) + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let gelu = gelu_table();
    match index {
        // The ctest fixture: mixed exponents, ungated and gated.
        0 | 1 => {
            let aexp = [-3, 0, 5, -40, 120, -100, 10, -10];
            let bexp = [-2, 1, -1, 15, 100, -100, 120, -2];
            build(rng, entries, 8, 32, index == 1, 0, 0, 0, &gelu, |_, row, _, gate| {
                let mut codes = [0i8; 32];
                for (i, code) in codes.iter_mut().enumerate() {
                    let (i, row) = (i as i32, row as i32);
                    *code = if gate {
                        if row == 7 {
                            0
                        } else {
                            ((i * 19 + row * 3) % 127 - 63) as i8
                        }
                    } else if row == 6 {
                        (-(i % 63 + 1)) as i8
                    } else {
                        ((i * 13 + row) % 127 - 63) as i8
                    };
                }
                (if gate { bexp[row] } else { aexp[row] }, codes)
            })
        }
        // Every code at every exponent from -30 to 10.
        2 => build(rng, entries, 41, 256, false, 3, 5, 0, &gelu, |_, row, b, _| {
            let mut codes = [0i8; 32];
            for (lane, code) in codes.iter_mut().enumerate() {
                let i = b * 32 + lane;
                *code = if i < 255 { (i as i32 - 127) as i8 } else { 0 };
            }
            (row as i32 - 30, codes)
        }),
        // Every gate code against swept activation and gate exponents.
        3 => build(rng, entries, 20, 256, true, 0, 0, 0, &gelu, |_, row, b, gate| {
            let mut codes = [0i8; 32];
            for (lane, code) in codes.iter_mut().enumerate() {
                let i = (b * 32 + lane) as i32;
                *code = if gate {
                    (i % 255 - 127) as i8
                } else {
                    ((i * 37 + row as i32) % 255 - 127) as i8
                };
            }
            (
                if gate {
                    (row as i32 * 7) % 41 - 20
                } else {
                    row as i32 - 14
                },
                codes,
            )
        }),
        // Extreme and boundary input exponents.
        4 => {
            let list = [
                -127, -126, -100, -26, -25, -24, -23, -12, -11, -10, -9, -1, 0, 3, 4, 5, 6, 30, 100, 126, 127,
            ];
            build(rng, entries, 16, 256, false, 0, 0, 0, &gelu, |rng, row, b, _| {
                (list[(row * 8 + b) % list.len()], random_codes(rng))
            })
        }
        // Gated scales that clamp the output exponent high or low.
        5 => {
            let pairs = [
                (100, 60),
                (127, 127),
                (126, 100),
                (-100, -100),
                (-127, -127),
                (-26, -120),
                (5, -127),
                (-10, 127),
            ];
            build(rng, entries, 16, 256, true, 0, 0, 0, &gelu, |rng, row, b, gate| {
                let (a, g) = pairs[(row * 8 + b) % pairs.len()];
                (if gate { g } else { a }, random_codes(rng))
            })
        }
        // Zero, single-lane and tiny blocks.
        6 => build(rng, entries, 4, 512, false, 0, 0, 0, &gelu, |rng, _, b, _| {
            let mut codes = [0i8; 32];
            match b % 4 {
                0 => {}
                1 => codes[rng.below(32) as usize] = rng.code(),
                2 => codes.iter_mut().for_each(|code| *code = [-1, 1][rng.below(2) as usize]),
                _ => codes = random_codes(rng),
            }
            (random_exponent(rng), codes)
        }),
        // One full scale row, random table.
        7 => {
            let table = random_table(rng);
            build(
                rng,
                entries,
                2,
                512,
                false,
                7,
                100,
                entries - 1024,
                &table,
                |rng, _, _, _| random_block(rng),
            )
        }
        // A partial second scale row, gated.
        8 => build(
            rng,
            entries,
            3,
            544,
            true,
            1,
            2,
            entries - 1024,
            &gelu,
            |rng, _, _, _| random_block(rng),
        ),
        // The widest row that fits the bank, ungated and gated, ending at the bank top.
        9 | 14 => {
            let parts = 1 + usize::from(index == 14);
            let k = (1..)
                .map(|blocks| blocks * 32)
                .take_while(|&k| row_stride(k) * parts <= entries)
                .last()
                .unwrap();
            build(
                rng,
                entries,
                1,
                k,
                index == 14,
                entries - row_stride(k) * parts,
                0,
                0,
                &gelu,
                |rng, _, _, _| random_block(rng),
            )
        }
        // The most rows that fit, ungated and gated, ending at the bank top.
        10 => {
            let rows = entries / 3;
            build(
                rng,
                entries,
                rows,
                32,
                false,
                entries - rows * 3,
                0,
                0,
                &gelu,
                |rng, _, _, _| random_block(rng),
            )
        }
        11 => {
            let rows = entries / 6;
            build(
                rng,
                entries,
                rows,
                32,
                true,
                entries - rows * 6,
                entries - rows * 3,
                0,
                &gelu,
                |rng, _, _, _| random_block(rng),
            )
        }
        // Completion held back by the host.
        12 => {
            let mut case = build(rng, entries, 2, 64, false, 0, 0, 0, &gelu, |rng, _, _, _| {
                random_block(rng)
            });
            case.hold = 300;
            case
        }
        // Interpolation-heavy exponents on a random table.
        13 => {
            let table = random_table(rng);
            build(rng, entries, 8, 128, true, 0, 0, 0, &table, |rng, _, _, gate| {
                (
                    if gate {
                        random_exponent(rng)
                    } else {
                        rng.below(11) as i32 - 6
                    },
                    random_codes(rng),
                )
            })
        }
        _ => {
            let gated = rng.below(2) == 1;
            let blocks = 1 + rng.below(40) as usize;
            let k = blocks * 32;
            let stride = row_stride(k);
            let parts = 1 + usize::from(gated);
            let rows = 1 + rng.below((entries / (stride * parts)).min(256 / blocks) as u64) as usize;
            let in_base = rng.below((entries - rows * stride * parts + 1) as u64) as usize;
            let out_base = rng.below((entries - rows * stride + 1) as u64) as usize;
            let table_base = rng.below((entries - 1024 + 1) as u64) as usize;
            let table = if rng.below(10) < 3 { random_table(rng) } else { gelu };
            build(
                rng,
                entries,
                rows,
                k,
                gated,
                in_base,
                out_base,
                table_base,
                &table,
                |rng, _, _, _| random_block(rng),
            )
        }
    }
}
