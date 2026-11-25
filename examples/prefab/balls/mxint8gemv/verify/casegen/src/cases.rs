// Directed and random MXINT8GEMV cases. Each case is a set of 1024-row bank
// images: role 0 = A, 1..4 = B groups 0..3, 5 = old C, 6 = new C before the
// instruction, 7 = new C after it.
use crate::reference::{self, Command, Flags, State};
use std::sync::Mutex;

const DIRECTED: usize = 18;
const RANDOM: usize = 40;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u32) as i32
    }
}

// Logical operands of one instruction: A codes/exponents and 32 B channels.
struct Data {
    a: Vec<i8>,
    ae: Vec<u8>,
    b: Vec<Vec<i8>>,
    be: Vec<Vec<u8>>,
}

// codes: 0 full INT8, 1 small [-3,3], 2 sparse. exps: 0 near 127, 1 full
// [0,254], 2 per-block jumps.
fn code(rng: &mut Rng, kind: u32) -> i8 {
    match kind {
        0 => rng.range(-128, 127) as i8,
        1 => rng.range(-3, 3) as i8,
        _ => if rng.next() % 5 == 0 { rng.range(-128, 127) as i8 } else { 0 },
    }
}
fn exponent(rng: &mut Rng, kind: u32, block: usize) -> u8 {
    match kind {
        0 => rng.range(119, 135) as u8,
        1 => rng.range(0, 254) as u8,
        _ => (rng.range(-6, 6) + [127, 60, 200, 10, 250][block % 5]).clamp(0, 254) as u8,
    }
}
fn fill(rng: &mut Rng, k: usize, codes: u32, exps: u32) -> Data {
    let blocks = k / 32;
    let a = (0..k).map(|_| code(rng, codes)).collect();
    let ae = (0..blocks).map(|n| exponent(rng, exps, n)).collect();
    let b = (0..32).map(|_| (0..k).map(|_| code(rng, codes)).collect()).collect();
    let be = (0..32).map(|_| (0..blocks).map(|n| exponent(rng, exps, n)).collect()).collect();
    Data { a, ae, b, be }
}

// The state an earlier instruction of the same sequence left behind.
fn previous(rng: &mut Rng, blocks: usize, zero: bool, codes: u32, exps: u32) -> State {
    let mut state = State { values: [0; 32], scale: 0, valid: false, processed: blocks };
    let mut flags = Flags::default();
    for _ in 0..blocks {
        let mut data = fill(rng, 32, codes, exps);
        if zero {
            data.a.fill(0);
        }
        let lanes: Vec<&[i8]> = data.b.iter().map(|w| &w[..]).collect();
        let be: Vec<u8> = data.be.iter().map(|e| e[0]).collect();
        reference::block(&mut state, &data.a, data.ae[0], &lanes, &be, &mut flags);
    }
    state
}

pub struct Case {
    pub fields: Vec<u32>,
    pub words: Vec<Vec<u128>>,
}

fn make(index: usize) -> Case {
    let mut rng = Rng(0x9e3779b97f4a7c15 ^ (index as u64 * 0x100000001b3));
    let directed_k = [32, 32, 512, 1024, 64, 512, 96, 64, 96, 128, 64, 32, 64, 32, 1984, 1024, 768, 32];
    let k = if index < DIRECTED {
        directed_k[index]
    } else if rng.next() % 4 == 0 {
        32 * rng.range(33, 62) as usize
    } else {
        32 * rng.range(1, 32) as usize
    };
    let blocks = k / 32;
    let stride = reference::stride(k);
    let (codes, exps) = if index < DIRECTED { (0, 0) } else { (rng.next() % 3, rng.next() % 3) };
    let mut d = fill(&mut rng, k, codes, exps);
    // accumulate, finalize, previous blocks, zero previous state, in-place C.
    let (accumulate, finalize, prev, prev_zero, inplace) = match index {
        1 | 3 | 11 | 12 | 13 | 17 => (false, true, 0, false, false),
        4 => (true, false, 3, false, false),
        5 => (true, true, 16, false, true),
        7 => (true, true, 5, true, false),
        15 => (true, false, 992, false, false),
        16 => (true, true, 1000, false, false),
        0..DIRECTED => (false, false, 0, false, false),
        _ => {
            let accumulate = rng.next() % 2 == 0;
            let room = 1024 - blocks;
            let prev = if rng.next() % 4 == 0 { rng.range(1, room as i32) } else { rng.range(1, 8) };
            (accumulate, rng.next() % 2 == 0, prev as usize, rng.next() % 8 == 0, accumulate && rng.next() % 4 == 0)
        }
    };
    let prev = if accumulate { prev } else { 0 };
    match index {
        0 | 1 => {
            for i in 0..32 {
                d.a[i] = i as i8 - 16;
                for j in 0..32 {
                    d.b[j][i] = ((j * 3 + i) % 9) as i8 - 4;
                }
            }
            d.ae[0] = 127;
            for j in 0..32 {
                d.be[j][0] = 120 + (j % 8) as u8;
            }
        }
        3 => d = fill(&mut rng, k, 0, 2),
        6 => d.a.fill(0),
        7 => d.a.fill(0),
        8 => {
            // Block 1 cancels block 0 exactly; block 2 starts a fresh scale.
            for i in 0..32 {
                d.a[i] = d.a[i].max(-127);
                d.a[32 + i] = -d.a[i];
            }
            d.ae[1] = d.ae[0];
            d.ae[2] = 140;
            for j in 0..32 {
                d.b[j].copy_within(0..32, 32);
                d.be[j][1] = d.be[j][0];
            }
        }
        9 => {
            // Scale -254, then 254 (state flush), then -254 again (term flush).
            d.ae[..3].copy_from_slice(&[0, 254, 0]);
            d.ae[3] = 127;
            for j in 0..32 {
                d.be[j][..3].copy_from_slice(&[0, 254, 0]);
                d.be[j][3] = 127;
            }
        }
        10 => {
            // Odd products shifted by one and by two produce signed ties.
            d.a.fill(0);
            d.a[0] = 1;
            d.a[32] = 1;
            d.ae[..2].copy_from_slice(&[127, 127]);
            for j in 0..32 {
                d.b[j].fill(0);
                d.b[j][0] = (j as i32 * 2 - 31) as i8;
                d.b[j][32] = (j as i32 * 2 - 31) as i8;
                d.be[j][..2].copy_from_slice(&[127, 127]);
            }
            d.be[0][..2].copy_from_slice(&[160, 161]);
        }
        11 => {
            d.ae[0] = 254;
            for j in 0..32 {
                d.be[j][0] = 254;
                if j % 5 == 0 {
                    d.b[j].fill(0);
                }
            }
        }
        12 => {
            // Small lanes take the exact left-shift path, large ones saturate.
            d.a.fill(0);
            d.a[0] = 16;
            d.a[32] = 100;
            d.ae[..2].copy_from_slice(&[127, 161]);
            for j in 0..32 {
                d.b[j].fill(0);
                d.b[j][0] = (j as i32 * 8 - 128) as i8;
                d.be[j][..2].copy_from_slice(&[252, 254]);
            }
            d.b[0][0] = 0;
            d.b[0][32] = 100;
        }
        13 => {
            d.ae[0] = 0;
            for j in 0..32 {
                d.be[j][0] = 0;
            }
        }
        14 => {
            d.a.fill(-128);
            d.ae.fill(127);
            for j in 0..32 {
                d.b[j].fill(-128);
                d.be[j].fill(127);
            }
        }
        17 => {
            // Finalize ties: +-127.5 rounds to +-128 and clamps, 0.5 to 0.
            d.a.fill(0);
            d.a[0] = 15;
            d.a[1] = 1;
            d.ae[0] = 127;
            let lanes = [(17, 0), (-17, 0), (0, 1), (0, 3), (0, -3), (0, -1), (0, 5), (0, 7)];
            for j in 0..32 {
                d.b[j].fill(0);
                (d.b[j][0], d.b[j][1]) = lanes[j % 8];
                d.be[j][0] = 127;
            }
        }
        _ => {}
    }
    let (a_codes, a_exp, b_base, c_base) = match index {
        2 => (2, 1000, 3, 0),
        14 => (896, 1020, 0, 1007),
        0..DIRECTED => (2, 1000, 3, 5),
        _ => {
            let a_codes = rng.range(0, (1024 - k / 16) as i32) as usize;
            let mut a_exp;
            loop {
                a_exp = rng.range(0, (1024 - blocks.div_ceil(16)) as i32) as usize;
                if a_exp + blocks.div_ceil(16) <= a_codes || a_exp >= a_codes + k / 16 {
                    break;
                }
            }
            let b_base = if rng.next() % 4 == 0 { 0 } else { rng.range(0, (1024 - 8 * stride) as i32) as usize };
            (a_codes, a_exp, b_base, rng.range(0, 1007) as usize)
        }
    };
    // op1 bank, op2 bank, wr bank, wr read bank, op1/op2/wr groups.
    let banks: [u32; 7] = match index {
        5 => [1, 2, 3, 3, 1, 2, 1],
        14 => [0, 15, 7, 9, 23, 20, 0],
        0..DIRECTED => [1, 2, 3, 4, 1, 2, 1],
        _ => {
            let mut ids = [0u32; 4];
            for n in 0..4 {
                loop {
                    ids[n] = rng.range(0, 15) as u32;
                    if !ids[..n].contains(&ids[n]) {
                        break;
                    }
                }
            }
            if inplace {
                ids[3] = ids[2];
            }
            [ids[0], ids[1], ids[2], ids[3], rng.range(0, 23) as u32, rng.range(0, 20) as u32, rng.range(0, 23) as u32]
        }
    };
    let junk = |rng: &mut Rng| (0..16384).map(|_| rng.next() as u8).collect::<Vec<u8>>();
    let mut a = junk(&mut rng);
    for i in 0..k {
        a[a_codes * 16 + i] = d.a[i] as u8;
    }
    for n in 0..blocks {
        a[a_exp * 16 + n] = d.ae[n];
    }
    let mut b: Vec<Vec<u8>> = (0..4).map(|_| junk(&mut rng)).collect();
    for j in 0..32 {
        let row = &mut b[j / 8][(b_base + j % 8 * stride) * 16..(b_base + j % 8 * stride + stride) * 16];
        row.fill(0);
        for i in 0..k {
            row[i] = d.b[j][i] as u8;
        }
        row[k..k + blocks].copy_from_slice(&d.be[j]);
    }
    let mut c_old = junk(&mut rng);
    let mut old_valid = false;
    if accumulate {
        let state = previous(&mut rng, prev, prev_zero, codes, exps);
        old_valid = state.valid;
        c_old[c_base * 16..c_base * 16 + 272].copy_from_slice(&reference::encode_state(&state));
    }
    let c_initial = if inplace { c_old.clone() } else { vec![0x5a; 16384] };
    let mut c_new = c_initial.clone();
    let cmd = Command { k, a_codes, a_exp, b_base, c_base, accumulate, finalize };
    let mut flags = Flags::default();
    reference::execute(&cmd, &a, &b, &c_old, &mut c_new, &mut flags);
    let f = [
        flags.align,
        flags.align_flush,
        flags.term_round,
        flags.term_flush,
        flags.tie,
        flags.zero_block,
        flags.restart,
        flags.q_high,
        flags.q_low,
        flags.code_sat,
        flags.round_clamp,
        flags.left_shift,
        flags.zero_out,
        flags.edge_code,
        prev + blocks == 1024,
        accumulate && old_valid,
        accumulate && !old_valid,
        inplace,
    ];
    let mut fields = vec![k as u32, a_codes as u32, a_exp as u32, b_base as u32, c_base as u32];
    fields.extend([accumulate as u32, finalize as u32]);
    fields.extend(banks);
    fields.extend(f.iter().map(|x| *x as u32));
    let words = [&a, &b[0], &b[1], &b[2], &b[3], &c_old, &c_initial, &c_new]
        .iter()
        .map(|image| image.chunks(16).map(|row| u128::from_le_bytes(row.try_into().unwrap())).collect())
        .collect();
    Case { fields, words }
}

static CURRENT: Mutex<Option<Case>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn mxint8_case_count() -> u32 {
    (DIRECTED + RANDOM) as u32
}
#[no_mangle]
pub extern "C" fn mxint8_case_load(index: u32) {
    assert!((index as usize) < DIRECTED + RANDOM, "Invalid case index");
    *CURRENT.lock().unwrap() = Some(make(index as usize));
}
#[no_mangle]
pub extern "C" fn mxint8_case_field(field: u32) -> u32 {
    CURRENT.lock().unwrap().as_ref().unwrap().fields[field as usize]
}
#[no_mangle]
pub extern "C" fn mxint8_case_word(role: u32, row: u32, high: u32) -> u64 {
    (CURRENT.lock().unwrap().as_ref().unwrap().words[role as usize][row as usize] >> (64 * high)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn directed_cases_hit_their_paths() {
        let mut flags = Flags::default();
        assert_eq!(reference::round_shift(5, 1, &mut flags), 2);
        assert_eq!(reference::round_shift(7, 1, &mut flags), 4);
        assert_eq!(reference::round_shift(-5, 1, &mut flags), -2);
        assert_eq!(reference::round_shift(-7, 1, &mut flags), -4);
        let hit = |index: usize, field: usize| make(index).fields[14 + field] == 1;
        assert!(hit(6, 5) && hit(8, 6) && hit(9, 1) && hit(9, 3) && hit(10, 4));
        assert!(hit(11, 7) && hit(11, 9) && hit(12, 11) && hit(13, 8) && hit(7, 12));
        assert!(hit(14, 13) && hit(15, 14) && hit(16, 14) && hit(17, 10) && hit(5, 17));
        let case = make(17);
        let row = case.words[7][5].to_le_bytes();
        assert_eq!(&row[..8], &[127, 129, 0, 2, 254, 0, 2, 4]);
        for index in 0..DIRECTED + RANDOM {
            make(index);
        }
    }
}
