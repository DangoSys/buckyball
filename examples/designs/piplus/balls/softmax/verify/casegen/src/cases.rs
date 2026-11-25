//! SOFTMAX cases: the command fields, the input bank image and the expected
//! output bank image, handed to the testbench through DPI.
use crate::reference;
use std::cell::RefCell;

/// Cases 0 to 17 are directed, the others random.
const CASES: u32 = 50;

struct Case {
    /// rows, w, in_base, out_base, scale_log2, then the events of
    /// reference::execute: underflow, -inf, x - m overflow, max-tie rows,
    /// subnormal x, subnormal y, u near -126.
    fields: [u32; 12],
    images: [Vec<u8>; 2],
}
thread_local! { static CASE: RefCell<Option<Case>> = const { RefCell::new(None) }; }

fn bank_entries() -> usize {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn pow2(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

/// A row of inputs whose scaled values x = in * 2^k mix -inf, zeros, ties and
/// uniform values of a random range, with at least one finite value.
fn random_row(rng: &mut Rng, w: usize, k: i32) -> Vec<f32> {
    let range = [1.0f32, 8.0, 64.0, 300.0][(rng.next() % 4) as usize];
    let mut x = Vec::with_capacity(w);
    for i in 0..w {
        let pick = rng.next() % 100;
        let v = if pick < 5 {
            f32::NEG_INFINITY
        } else if pick < 10 {
            0.0
        } else if pick < 15 && i > 0 {
            x[(rng.next() as usize) % i]
        } else {
            rng.signed() * range
        };
        x.push(v);
    }
    if x.iter().all(|v| *v == f32::NEG_INFINITY) {
        x[(rng.next() as usize) % w] = 0.5;
    }
    x.iter().map(|&v| v * pow2(-k)).collect()
}

#[no_mangle]
pub extern "C" fn softmax_case_count() -> u32 {
    CASES
}

#[no_mangle]
pub extern "C" fn softmax_case_load(index: u32) {
    assert!(index < CASES);
    let entries = bank_entries();
    let mut rng = Rng(0x9e37_79b9 ^ index.wrapping_mul(0x85eb_ca6b));
    let (rows, w, in_base, out_base, k, inputs): (usize, usize, usize, usize, i32, Vec<Vec<f32>>) = match index {
        0 => (1, 16, 0, 0, 0, vec![(0..16).map(|i| i as f32 / 4.0 - 2.0).collect()]),
        1 => (3, 18, 0, 0, 0, (0..3).map(|_| random_row(&mut rng, 18, 0)).collect()),
        2 => (2, 17, 5, 100, 0, (0..2).map(|_| random_row(&mut rng, 17, 0)).collect()),
        3 => (4, 19, 0, 200, 0, (0..4).map(|_| random_row(&mut rng, 19, 0)).collect()),
        // Both banks full: rows of the widest row.
        4 => {
            let rows = entries / 512;
            (
                rows,
                2048,
                0,
                0,
                0,
                (0..rows).map(|_| random_row(&mut rng, 2048, 0)).collect(),
            )
        }
        // Equal values: every output is 1 / 64.
        5 => (
            4,
            64,
            7,
            3,
            0,
            [1.5f32, 0.0, -7.0, -3.0e38].iter().map(|&v| vec![v; 64]).collect(),
        ),
        // One finite value (an output 1.0) or many -inf.
        6 => (
            3,
            20,
            0,
            0,
            0,
            vec![
                (0..20).map(|i| if i == 7 { -3.5 } else { f32::NEG_INFINITY }).collect(),
                (0..20)
                    .map(|i| if i % 2 == 0 { f32::NEG_INFINITY } else { i as f32 * 0.75 })
                    .collect(),
                (0..20)
                    .map(|i| if i == 19 { 100.0 } else { f32::NEG_INFINITY })
                    .collect(),
            ],
        ),
        // x - m overflows to -inf; most exponentials underflow; u around -126.
        7 => {
            let edge = -126.0f32 / reference::LOG2E;
            let mut near: Vec<f32> = (0..32).map(|i| -86.0 - i as f32 * 0.08).collect();
            near[0] = 0.0;
            near[1] = edge;
            near[2] = f32::from_bits(edge.to_bits() - 1);
            near[3] = f32::from_bits(edge.to_bits() + 1);
            (
                3,
                32,
                0,
                0,
                0,
                vec![
                    (0..32).map(|i| [3.0e38f32, -3.0e38, 1.0e38, -1.0e38][i % 4]).collect(),
                    (0..32).map(|i| -(i as f32) * 3125.0).collect(),
                    near,
                ],
            )
        }
        // The extreme scales.
        8 => (
            2,
            16,
            0,
            0,
            127,
            (0..2)
                .map(|_| (0..16).map(|_| rng.signed() * pow2(-124)).collect())
                .collect(),
        ),
        9 => (
            2,
            16,
            0,
            0,
            -126,
            (0..2)
                .map(|_| (0..16).map(|_| rng.signed() * 2.0 * pow2(126)).collect())
                .collect(),
        ),
        10 => (2, 24, 0, 0, 5, (0..2).map(|_| random_row(&mut rng, 24, 5)).collect()),
        11 => (2, 24, 0, 0, -7, (0..2).map(|_| random_row(&mut rng, 24, -7)).collect()),
        // Zeros of both signs and subnormal inputs.
        12 => (
            2,
            40,
            0,
            0,
            0,
            vec![
                (0..40)
                    .map(|i| {
                        [
                            0.0f32,
                            -0.0,
                            1.0e-40,
                            -1.0e-42,
                            f32::MIN_POSITIVE,
                            1.0e-39,
                            -2.5e-41,
                            1.0e-45,
                        ][i % 8]
                    })
                    .collect(),
                (0..40).map(|i| if i % 3 == 0 { -0.0 } else { 0.0 }).collect(),
            ],
        ),
        // Ties of the maximum.
        13 => (
            2,
            36,
            0,
            0,
            0,
            (0..2)
                .map(|r| {
                    (0..36)
                        .map(|i| [2.0f32, -1.0, 0.5, 2.0, -3.25, 2.0][(i + r) % 6])
                        .collect()
                })
                .collect(),
        ),
        // Many short rows filling the input bank.
        14 => {
            let rows = entries / 4;
            (
                rows,
                16,
                0,
                0,
                0,
                (0..rows).map(|_| random_row(&mut rng, 16, 0)).collect(),
            )
        }
        // A row ending on the last bank row.
        15 => (1, 2047, entries - 512, 1, 0, vec![random_row(&mut rng, 2047, 0)]),
        // Completion back-pressure (the testbench holds cmdResp.ready).
        16 => (1, 16, 0, 0, 0, vec![random_row(&mut rng, 16, 0)]),
        // Subnormal outputs: e near 2^-126 divided by s near 4.
        17 => (
            1,
            16,
            0,
            0,
            0,
            vec![(0..16)
                .map(|i| if i < 4 { 0.0 } else { -87.0 - (i - 4) as f32 * 0.03 })
                .collect()],
        ),
        _ => {
            let w = match rng.next() % 3 {
                0 => 16 + (rng.next() % 49) as usize,
                1 => 16 + (rng.next() % 497) as usize,
                _ => 16 + (rng.next() % 2033) as usize,
            };
            let lines = w.div_ceil(4);
            let most = entries / lines;
            let rows = if rng.next() % 4 == 0 {
                most
            } else {
                1 + (rng.next() as usize) % most.min(8)
            };
            let in_base = (rng.next() as usize) % (entries - rows * lines + 1);
            let out_base = (rng.next() as usize) % (entries - rows * lines + 1);
            let k = (rng.next() % 41) as i32 - 20;
            (
                rows,
                w,
                in_base,
                out_base,
                k,
                (0..rows).map(|_| random_row(&mut rng, w, k)).collect(),
            )
        }
    };
    let lines = w.div_ceil(4);
    // Unused rows hold NaN; the lanes after w of a row's last bank row hold
    // inf, NaN and -inf, which the Ball must ignore.
    let mut input: Vec<u8> = (0..entries * 4).flat_map(|_| 0x7fc1_2345u32.to_le_bytes()).collect();
    for (row, values) in inputs.iter().enumerate() {
        for lane in 0..lines * 4 {
            let bits = if lane < w {
                values[lane].to_bits()
            } else {
                [0x7f80_0000u32, 0x7fc0_0001, 0xff80_0000][lane % 3]
            };
            let at = ((in_base + row * lines) * 4 + lane) * 4;
            input[at..at + 4].copy_from_slice(&bits.to_le_bytes());
        }
    }
    let mut output = vec![0x5au8; entries * 16];
    let events = reference::execute(&input, &mut output, rows, w, in_base, out_base, k);
    let mut fields = [
        rows as u32,
        w as u32,
        in_base as u32,
        out_base as u32,
        k as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    fields[5..].copy_from_slice(&events);
    CASE.with(|c| {
        *c.borrow_mut() = Some(Case {
            fields,
            images: [input, output],
        })
    });
}

#[no_mangle]
pub extern "C" fn softmax_case_field(field: u32) -> u32 {
    CASE.with(|c| c.borrow().as_ref().unwrap().fields[field as usize])
}

/// Half `high` of bank row `row` of the input (role 0) or expected output (role 1) image.
#[no_mangle]
pub extern "C" fn softmax_case_word(role: u32, row: u32, high: u32) -> u64 {
    CASE.with(|c| {
        let c = c.borrow();
        let bytes = &c.as_ref().unwrap().images[role as usize];
        let at = row as usize * 16 + high as usize * 8;
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    })
}

#[no_mangle]
pub extern "C" fn softmax_bank_entries() -> u32 {
    bank_entries() as u32
}
