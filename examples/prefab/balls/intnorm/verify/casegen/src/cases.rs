use crate::reference::{self, Stats};
use std::cell::RefCell;
use std::ffi::{c_char, CString};

const CASES: u32 = 23;

struct Case {
    name: CString,
    rows: usize,
    k: usize,
    // Bank rows of 16 bytes.
    input_base: usize,
    coefficient_base: usize,
    output_base: usize,
    epsilon: u64,
    // Cycles the test holds cmdResp.ready low once the Ball completes.
    stall: u32,
    entries: usize,
    // Input, coefficients, output before and output after the command.
    images: [Vec<u8>; 4],
    scratch: Vec<Vec<i32>>,
    stats: Stats,
}

thread_local! { static CASE: RefCell<Option<Case>> = const { RefCell::new(None) }; }

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
    fn signed(&mut self, magnitude: i64) -> i64 {
        self.below(2 * magnitude as u64 + 1) as i64 - magnitude
    }
    // A uniform FP32 value in [-2^scale, 2^scale) plus an offset.
    fn real(&mut self, scale: i32, offset: f64) -> u32 {
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (((unit * 2.0 - 1.0) * 2f64.powi(scale) + offset) as f32).to_bits()
    }
}

fn config(key: &str) -> usize {
    let path = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(path).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("{key} missing from BB_VERIFY_CONFIG"))
        .parse()
        .unwrap()
}

// Footprints of one command in bank rows: input, coefficients, output with scratch.
fn footprint(rows: usize, k: usize) -> (usize, usize, usize) {
    (rows * k / 4, 1 + k / 2, rows * (k / 16 + k / 512) + k / 4)
}

#[allow(clippy::too_many_arguments)]
fn build(
    name: &str,
    entries: usize,
    rows: usize,
    k: usize,
    bases: [usize; 3],
    epsilon: u64,
    stall: u32,
    input: impl Fn(usize, usize) -> u32,
    gamma: impl Fn(usize) -> i32,
    beta: impl Fn(usize) -> i32,
) -> Case {
    let (input_rows, coefficient_rows, output_rows) = footprint(rows, k);
    assert!(bases[0] + input_rows <= entries, "{name}: input exceeds the bank");
    assert!(bases[1] + coefficient_rows <= entries, "{name}: coefficients exceed the bank");
    assert!(bases[2] + output_rows <= entries, "{name}: output exceeds the bank");
    // Rows outside the footprint hold NaN inputs, junk coefficients and a
    // guard pattern, so a stray access shows up as an assertion or a mismatch.
    let mut images = [
        0x7fc12345u32.to_le_bytes().repeat(entries * 4),
        vec![0xa5; entries * 16],
        vec![0x5a; entries * 16],
        Vec::new(),
    ];
    for row in 0..rows {
        for col in 0..k {
            let at = bases[0] * 16 + (row * k + col) * 4;
            images[0][at..at + 4].copy_from_slice(&input(row, col).to_le_bytes());
        }
    }
    let at = bases[1] * 16;
    images[1][at..at + 8].copy_from_slice(&epsilon.to_le_bytes());
    images[1][at + 8..at + 16].fill(0);
    for col in 0..k {
        let g = at + 16 + col * 4;
        let b = at + 16 + (k + col) * 4;
        images[1][g..g + 4].copy_from_slice(&gamma(col).to_le_bytes());
        images[1][b..b + 4].copy_from_slice(&beta(col).to_le_bytes());
    }
    let mut after = images[2].clone();
    let mut stats = Stats::default();
    let scratch = reference::execute(
        &images[0],
        &images[1],
        &mut after,
        rows,
        k,
        bases[0] * 16,
        bases[1] * 16,
        bases[2] * 16,
        &mut stats,
    );
    images[3] = after;
    Case {
        name: CString::new(name).unwrap(),
        rows,
        k,
        input_base: bases[0],
        coefficient_base: bases[1],
        output_base: bases[2],
        epsilon,
        stall,
        entries,
        images,
        scratch,
        stats,
    }
}

// A random command that fits the bank, with rows of random scale and mean.
fn random(index: u32, entries: usize, stall: u32) -> Case {
    let mut rng = Rng(0x9e3779b97f4a7c15 ^ u64::from(index).wrapping_mul(0x2545f4914f6cdd1d));
    let k = [512, 1024, 1536][rng.below(3) as usize];
    let (input_rows, _, output_rows) = footprint(1, k);
    let most = (entries / input_rows).min((entries - k / 4) / (output_rows - k / 4));
    let rows = 1 + rng.below(most as u64) as usize;
    let (input_rows, coefficient_rows, output_rows) = footprint(rows, k);
    let bases = [
        rng.below((entries - input_rows + 1) as u64) as usize,
        rng.below((entries - coefficient_rows + 1) as u64) as usize,
        rng.below((entries - output_rows + 1) as u64) as usize,
    ];
    let width = rng.below(45);
    let epsilon = 1 + rng.below(1u64 << width);
    let scales: Vec<(i32, f64)> = (0..rows)
        .map(|_| {
            let scale = rng.below(26) as i32 - 12;
            (scale, rng.signed(1000) as f64 / 1000.0 * 2f64.powi(scale))
        })
        .collect();
    let values: Vec<u32> = (0..rows * k)
        .map(|i| {
            let (scale, offset) = scales[i / k];
            match rng.below(32) {
                0 => 0,
                1 => 0x80000000,
                2 => rng.below(0x800000) as u32 | (rng.below(2) as u32) << 31,
                _ => rng.real(scale, offset),
            }
        })
        .collect();
    let gamma_scale = 1i64 << (8 + rng.below(13));
    let beta_scale = 1i64 << rng.below(25);
    let gamma: Vec<i32> = (0..k).map(|_| rng.signed(gamma_scale) as i32).collect();
    let beta: Vec<i32> = (0..k).map(|_| rng.signed(beta_scale) as i32).collect();
    build(
        &format!("random{index}"),
        entries,
        rows,
        k,
        bases,
        epsilon,
        stall,
        |row, col| values[row * k + col],
        |col| gamma[col],
        |col| beta[col],
    )
}

fn case(index: u32, entries: usize) -> Case {
    let mut rng = Rng(0x5bd1e9955bd1e995 ^ u64::from(index + 1).wrapping_mul(0x9e3779b97f4a7c15));
    match index {
        // The workloads/ctests/intnorm_round.cpp example.
        0 => {
            let values = [-2.5f64, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -3.5];
            let case = build(
                "round",
                entries,
                1,
                512,
                [0, 0, 0],
                42950,
                0,
                |_, col| ((values[col % 8] / 65536.0) as f32).to_bits(),
                |_| 65536,
                |_| 0,
            );
            let codes = [-40i8, -40, 0, 0, 40, 40, 79, -79];
            for i in 0..512 {
                assert_eq!(case.images[3][i] as i8, codes[i % 8], "round: code {i}");
            }
            assert!(case.images[3][512..528].iter().all(|v| *v == 115), "round: exponents");
            case
        }
        // Inputs that all round to zero, so the result is beta.
        1 => {
            let zeros = [0u32, 0x80000000, 0x00000001, 0x807fffff, 0x33800000, 0x37000000, 0xb7000000];
            let gamma: Vec<i32> = (0..512).map(|_| rng.signed(1 << 20) as i32).collect();
            let beta: Vec<i32> = (0..512)
                .map(|col| match (col / 32) % 4 {
                    0 => 0,
                    1 => rng.signed(63) as i32,
                    2 => rng.signed(1 << 24) as i32,
                    _ => rng.signed(1) as i32,
                })
                .collect();
            build(
                "zero",
                entries,
                2,
                512,
                [3, 5, 7],
                1,
                0,
                |row, col| if row == 0 { zeros[col % 7] } else { 0 },
                |col| gamma[col],
                |col| beta[col],
            )
        }
        // The extreme Q16 inputs, then every exponent class.
        2 => {
            let values: Vec<u32> = (0..1024)
                .map(|i| {
                    if i < 512 {
                        match i % 4 {
                            0 => 0x46ffffff,
                            1 => 0xc7000000,
                            2 => 0xc6ffffff,
                            _ => rng.real(14, 0.0),
                        }
                    } else {
                        let exponent = rng.below(142) as u32;
                        (rng.below(2) as u32) << 31 | exponent << 23 | rng.below(0x800000) as u32
                    }
                })
                .collect();
            let gamma: Vec<i32> = (0..512).map(|_| rng.signed(1 << 20) as i32).collect();
            build(
                "extreme",
                entries,
                2,
                512,
                [0, 1, 2],
                rng.below(1 << 40) + 1,
                0,
                |row, col| values[row * 512 + col],
                |col| gamma[col],
                |_| 0,
            )
        }
        // Inputs exactly halfway between two Q16 steps.
        3 => {
            let values: Vec<u32> = (0..1536)
                .map(|col| {
                    if col % 32 == 31 {
                        return 0x37000000 | (rng.below(2) as u32) << 31;
                    }
                    let exponent = 111 + (col % 23) as u32;
                    let shift = 134 - exponent;
                    let mantissa = (rng.below(0x800000) as u32 >> shift << shift) | 1 << (shift - 1);
                    (rng.below(2) as u32) << 31 | exponent << 23 | (mantissa & 0x7fffff)
                })
                .collect();
            let gamma: Vec<i32> = (0..1536).map(|_| rng.signed(1 << 16) as i32).collect();
            let beta: Vec<i32> = (0..1536).map(|_| rng.signed(1 << 16) as i32).collect();
            build(
                "input_ties",
                entries,
                1,
                1536,
                [17, 33, 51],
                rng.below(1 << 20) + 1,
                0,
                |_, col| values[col],
                |col| gamma[col],
                |col| beta[col],
            )
        }
        // Constant rows give z = 0, so beta sets every MXINT8 block.
        4 => {
            let rows = [1.5f64, -7.25, 0.0];
            let beta: Vec<i32> = (0..512)
                .map(|col| {
                    let i = col % 32;
                    match col / 32 {
                        // Ties at both parities and a code that rounds to 128.
                        0 => [32767, -32767, 128 * (2 * i as i32 + 1), -128 * (2 * i as i32 + 1)][i % 4],
                        1 => [i32::MIN, rng.signed(1 << 30) as i32][(i != 0) as usize],
                        2 => [i32::MAX, -i32::MAX, rng.signed(1 << 30) as i32][(i % 3).min(2)],
                        3 => -1,
                        4 => 0,
                        5 => [127, -127, rng.signed(127) as i32][(i % 3).min(2)],
                        6 => [64, rng.signed(64) as i32][(i != 0) as usize],
                        7 => [-128, 2 * i as i32 + 1, -(2 * i as i32 + 1)][(i % 3).min(2)],
                        _ => {
                            let width = rng.below(31);
                            rng.signed(1i64 << width) as i32
                        }
                    }
                })
                .collect();
            build(
                "beta_codes",
                entries,
                3,
                512,
                [9, 11, 13],
                1 + rng.below(1000),
                0,
                |row, _| (rows[row] as f32).to_bits(),
                // Gamma is irrelevant when every z is zero.
                |_| -0x12345,
                |col| beta[col],
            )
        }
        // Every bank filled to its last row.
        5 => {
            let k = 512;
            let rows = entries / (k / 4);
            let (_, coefficient_rows, output_rows) = footprint(rows, k);
            let values: Vec<u32> = (0..rows * k)
                .map(|i| rng.real(i as i32 / k as i32 * 3 - 9, (i / k) as f64 - 3.0))
                .collect();
            let gamma: Vec<i32> = (0..k).map(|_| rng.signed(1 << 17) as i32).collect();
            let beta: Vec<i32> = (0..k).map(|_| rng.signed(1 << 20) as i32).collect();
            build(
                "capacity",
                entries,
                rows,
                k,
                [0, entries - coefficient_rows, entries - output_rows],
                1 + rng.below(1 << 30),
                0,
                |row, col| values[row * k + col],
                |col| gamma[col],
                |col| beta[col],
            )
        }
        6 | 7 => {
            let (name, rows, k, bases) = if index == 6 {
                ("k1024", 3, 1024, [100, 300, 400])
            } else {
                ("k1536", 2, 1536, [entries - 768, entries - 769, entries - 582])
            };
            let values: Vec<u32> = (0..rows * k).map(|i| rng.real(4, (i / k) as f64 * 5.0 - 4.0)).collect();
            let gamma: Vec<i32> = (0..k).map(|_| rng.signed(1 << 18) as i32).collect();
            let beta: Vec<i32> = (0..k).map(|_| rng.signed(1 << 22) as i32).collect();
            build(
                name,
                entries,
                rows,
                k,
                bases,
                1 + rng.below(1 << 24),
                0,
                |row, col| values[row * k + col],
                |col| gamma[col],
                |col| beta[col],
            )
        }
        // Epsilon dominates the variance; full-range gamma.
        8 => {
            let values: Vec<u32> = (0..1024)
                .map(|i| {
                    if i < 512 {
                        rng.real(6, 30.0)
                    } else {
                        [0x46fffe00, 0xc6fffe00][i % 2]
                    }
                })
                .collect();
            let gamma: Vec<i32> = (0..512).map(|i| [i32::MAX, -i32::MAX, i32::MIN][i % 3]).collect();
            let beta: Vec<i32> = (0..512).map(|_| rng.signed(1 << 28) as i32).collect();
            build(
                "epsilon",
                entries,
                2,
                512,
                [600, 700, 800],
                (1 << 62) + rng.below(1 << 40),
                0,
                |row, col| values[row * 512 + col],
                |col| gamma[col],
                |col| beta[col],
            )
        }
        // Negative gamma and beta close to the Q16 limits.
        9 => {
            let values: Vec<u32> = (0..1024).map(|_| rng.real(3, 0.0)).collect();
            let beta: Vec<i32> = (0..512).map(|i| [0x7000_0000, -0x7000_0000][i % 2]).collect();
            build(
                "affine",
                entries,
                2,
                512,
                [40, 400, 600],
                1 + rng.below(1 << 10),
                0,
                |row, col| values[row * 512 + col],
                |col| -(1 << 26) + col as i32,
                |col| beta[col],
            )
        }
        10 => random(index, entries, 400),
        _ => random(index, entries, 0),
    }
}

#[no_mangle]
pub extern "C" fn intnorm_case_count() -> u32 {
    CASES
}

#[no_mangle]
pub extern "C" fn intnorm_case_load(index: u32) {
    assert!(index < CASES, "case {index} out of range");
    let built = case(index, config("bank_entries"));
    CASE.with(|c| *c.borrow_mut() = Some(built));
}

#[no_mangle]
pub extern "C" fn intnorm_case_name() -> *const c_char {
    CASE.with(|c| c.borrow().as_ref().unwrap().name.as_ptr())
}

#[no_mangle]
pub extern "C" fn intnorm_case_field(field: u32) -> u64 {
    CASE.with(|c| {
        let c = c.borrow();
        let c = c.as_ref().unwrap();
        let s = &c.stats;
        [
            c.rows as u64,
            c.k as u64,
            c.input_base as u64,
            c.coefficient_base as u64,
            c.output_base as u64,
            u64::from(c.stall),
            c.entries as u64,
            config("ball_id") as u64,
            u64::from(s.input_ties),
            u64::from(s.code_ties),
            u64::from(s.clips),
            u64::from(s.zero_blocks),
            u64::from(s.narrow_blocks),
            u64::from(s.full_blocks),
            u64::from(s.negative_means),
            u64::from(s.epsilon_rows),
            u64::from(u64::BITS - c.epsilon.leading_zeros()),
        ][field as usize]
    })
}

// One 64-bit half of a bank row of an image.
#[no_mangle]
pub extern "C" fn intnorm_case_word(image: u32, row: u32, high: u32) -> u64 {
    CASE.with(|c| {
        let at = row as usize * 16 + high as usize * 8;
        u64::from_le_bytes(c.borrow().as_ref().unwrap().images[image as usize][at..at + 8].try_into().unwrap())
    })
}

// One 64-bit half of the scratch row `index` the Ball writes for `row`.
#[no_mangle]
pub extern "C" fn intnorm_case_scratch(row: u32, index: u32, high: u32) -> u64 {
    CASE.with(|c| {
        let c = c.borrow();
        let v = &c.as_ref().unwrap().scratch[row as usize];
        let at = index as usize * 4 + high as usize * 2;
        u64::from(v[at] as u32) | u64::from(v[at + 1] as u32) << 32
    })
}
