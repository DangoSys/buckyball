//! LAYERNORM cases for the UVM test: directed edges first, then random
//! instructions; each case gives the instruction fields and three bank images
//! (input, parameters, expected output) of 1024 rows of four FP32 lanes.
use crate::reference;
use std::cell::RefCell;

const ROWS: usize = 1024;
const DIRECTED: u32 = 23;
const CASES: u32 = DIRECTED + 40;
const GUARD: u32 = 0x5a5a_5a5a;

// Value classes; the test's covergroup bins them by number.
const RANDOM: u32 = 0;
const CONSTANT: u32 = 1;
const SUBNORMAL: u32 = 2;
const OFFSET: u32 = 3;
const ZEROS: u32 = 4;
const WIDE: u32 = 5;
const ZERO_EPS: u32 = 6;
const LARGE_EPS: u32 = 7;

struct Case {
    fields: [u32; 14],
    images: [Vec<u32>; 3],
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
    /// Uniform in [-1, 1) on a 2^-23 grid.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 8388608.0 - 1.0
    }
    /// A quiet NaN with a random payload: lanes and rows the Ball must not use.
    fn garbage(&mut self) -> u32 {
        0x7fc0_0000 | (self.next() as u32 & 0x803f_ffff)
    }
}

fn value(rng: &mut Rng, class: u32, row_scale: f32) -> f32 {
    match class {
        SUBNORMAL => f32::from_bits((rng.below(0x007f_ffff) as u32 + 1) | ((rng.next() as u32) & 0x8000_0000)),
        OFFSET => 1.0e4 + rng.unit(),
        ZEROS => [0.0, -0.0, 1.0, -1.0, 0.5, -2.0, 0.0, -0.0][rng.below(8) as usize],
        WIDE => rng.unit() * f32::from_bits(((rng.below(81) as u32) + 127 - 40) << 23),
        _ => rng.unit() * row_scale,
    }
}

fn parameter(rng: &mut Rng, class: u32) -> f32 {
    match class {
        ZEROS => [0.0, -0.0, 1.0, -1.0, 2.0 * rng.unit()][rng.below(5) as usize],
        _ => 2.0 * rng.unit(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    seed: u64,
    rows: usize,
    w: usize,
    in_base: usize,
    out_base: usize,
    param_base: usize,
    rms: bool,
    class: u32,
    banks: [u32; 6],
    backpressure: bool,
) -> Case {
    let lines = w.div_ceil(4);
    assert!(rows > 0 && (16..=2048).contains(&w));
    assert!(in_base + rows * lines <= ROWS && out_base + rows * lines <= ROWS);
    assert!(param_base + if rms { lines + 1 } else { 2 * lines + 1 } <= ROWS);
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut input: Vec<u32> = (0..ROWS * 4).map(|_| rng.garbage()).collect();
    let mut params: Vec<u32> = (0..ROWS * 4).map(|_| rng.garbage()).collect();
    for row in 0..rows {
        let row_scale = f32::from_bits((rng.below(17) as u32 + 127 - 8) << 23);
        let constant = if row == 0 { 0.0 } else { rng.unit() * row_scale };
        for i in 0..w {
            let v = if class == CONSTANT {
                constant
            } else {
                value(&mut rng, class, row_scale)
            };
            input[(in_base + row * lines) * 4 + i] = v.to_bits();
        }
    }
    for i in 0..w {
        params[param_base * 4 + i] = parameter(&mut rng, class).to_bits();
        if !rms {
            params[(param_base + lines) * 4 + i] = parameter(&mut rng, class).to_bits();
        }
    }
    let eps: f32 = match class {
        SUBNORMAL => 1.0e-30,
        ZERO_EPS => 0.0,
        LARGE_EPS => 1.0e6,
        _ => [1.0e-5, 1.0e-6, 1.0e-12][rng.below(3) as usize],
    };
    params[(param_base + if rms { lines } else { 2 * lines }) * 4] = eps.to_bits();
    let guard = vec![GUARD; ROWS * 4];
    let output = reference::expected(&input, &params, &guard, rows, w, in_base, out_base, param_base, rms);
    let fields = [
        rows as u32,
        w as u32,
        in_base as u32,
        out_base as u32,
        param_base as u32,
        rms as u32,
        banks[0],
        banks[1],
        banks[2],
        banks[3],
        banks[4],
        banks[5],
        class,
        backpressure as u32,
    ];
    Case {
        fields,
        images: [input, params, output],
    }
}

fn random(index: u32) -> Case {
    let mut rng = Rng(0x1234_5678_9abc_def0 ^ ((index as u64) << 32) ^ index as u64);
    let rms = rng.below(2) == 1;
    let w = match rng.below(4) {
        0 => 16 + rng.below(16),
        1 => 32 + rng.below(97),
        2 => 129 + rng.below(384),
        _ => 513 + rng.below(1532),
    } as usize;
    let lines = w.div_ceil(4);
    let rows = 1 + rng.below((ROWS / lines).min(48) as u64) as usize;
    let in_base = rng.below((ROWS - rows * lines + 1) as u64) as usize;
    let out_base = rng.below((ROWS - rows * lines + 1) as u64) as usize;
    let footprint = if rms { lines + 1 } else { 2 * lines + 1 };
    let param_base = rng.below((ROWS - footprint + 1) as u64) as usize;
    let class = [
        RANDOM, RANDOM, WIDE, OFFSET, ZEROS, CONSTANT, SUBNORMAL, ZERO_EPS, LARGE_EPS,
    ][rng.below(9) as usize];
    let mut banks = [0u32; 6];
    banks[0] = rng.below(24) as u32;
    banks[1] = (banks[0] + 1 + rng.below(23) as u32) % 24;
    banks[2] = loop {
        let b = rng.below(24) as u32;
        if b != banks[0] && b != banks[1] {
            break b;
        }
    };
    for group in &mut banks[3..] {
        *group = rng.below(24) as u32;
    }
    build(
        index as u64,
        rows,
        w,
        in_base,
        out_base,
        param_base,
        rms,
        class,
        banks,
        false,
    )
}

fn directed(index: u32) -> Case {
    let (rows, w, in_base, out_base, param_base, rms, class) = [
        (1, 16, 0, 0, 0, false, RANDOM),
        (1, 16, 0, 0, 0, true, RANDOM),
        (3, 17, 5, 40, 9, false, RANDOM),
        (2, 18, 0, 300, 7, true, RANDOM),
        (2, 19, 11, 0, 500, false, RANDOM),
        // The widest LayerNorm: parameters end on the last bank row.
        (1, 2044, 0, 513, 1, false, RANDOM),
        // The widest RMSNorm: input and parameters end on the last bank row.
        (1, 2048, 512, 0, 511, true, RANDOM),
        // Rows fill the whole input and output bank.
        (256, 16, 0, 0, 1015, false, RANDOM),
        (4, 64, 0, 64, 0, false, CONSTANT),
        (4, 63, 0, 64, 0, true, SUBNORMAL),
        (3, 33, 0, 100, 0, false, SUBNORMAL),
        (4, 128, 0, 200, 0, false, OFFSET),
        (4, 40, 0, 200, 0, false, ZEROS),
        (4, 40, 0, 200, 0, true, ZEROS),
        (3, 96, 0, 100, 0, false, WIDE),
        (3, 96, 0, 100, 0, true, WIDE),
        (2, 48, 0, 100, 0, false, ZERO_EPS),
        (2, 48, 0, 100, 0, false, LARGE_EPS),
        (3, 21, 7, 900, 300, true, CONSTANT),
        (2, 130, 600, 3, 17, true, OFFSET),
        (2, 22, 0, 100, 0, true, ZERO_EPS),
        (2, 23, 0, 100, 0, true, LARGE_EPS),
        // The completion waits on cmdResp backpressure.
        (2, 20, 0, 100, 0, true, RANDOM),
    ][index as usize];
    build(
        index as u64,
        rows,
        w,
        in_base,
        out_base,
        param_base,
        rms,
        class,
        [1, 2, 3, 1, 2, 3],
        index == DIRECTED - 1,
    )
}

#[no_mangle]
pub extern "C" fn layernorm_case_count() -> u32 {
    CASES
}

#[no_mangle]
pub extern "C" fn layernorm_case_load(index: u32) {
    assert!(index < CASES);
    let case = if index < DIRECTED {
        directed(index)
    } else {
        random(index)
    };
    CASE.with(|c| *c.borrow_mut() = Some(case));
}

/// 0 rows, 1 w, 2 in_base, 3 out_base, 4 param_base, 5 rms, 6..8 input,
/// params, output bank, 9..11 their groups, 12 value class, 13 backpressure.
#[no_mangle]
pub extern "C" fn layernorm_case_field(field: u32) -> u32 {
    CASE.with(|c| c.borrow().as_ref().unwrap().fields[field as usize])
}

/// Role 0 input, 1 parameters, 2 expected output: lanes 2 * high and
/// 2 * high + 1 of bank row `row`.
#[no_mangle]
pub extern "C" fn layernorm_case_word(role: u32, row: u32, high: u32) -> u64 {
    CASE.with(|c| {
        let c = c.borrow();
        let v = &c.as_ref().unwrap().images[role as usize];
        let at = row as usize * 4 + high as usize * 2;
        v[at] as u64 | ((v[at + 1] as u64) << 32)
    })
}
