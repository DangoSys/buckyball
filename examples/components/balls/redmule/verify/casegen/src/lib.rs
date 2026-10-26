#[path = "../../../emu/src/fp16.rs"]
mod fp16;
use std::cell::RefCell;

struct Case {
    rows: usize,
    k: usize,
    base: usize,
    blocks: usize,
    a: Vec<u16>,
    b: Vec<u16>,
    c: Vec<u16>,
}
thread_local! {
    static CASE: RefCell<Option<Case>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn redmule_case_load(index: u32, lanes: u32) {
    assert_eq!(lanes, 8);
    let rows = if index >= 64 {
        [16, 16, 32][index as usize - 64]
    } else {
        index as usize % 3 + 1
    };
    let k = if index >= 64 {
        [16, 32, 32][index as usize - 64]
    } else if index & 1 == 0 {
        8
    } else {
        16
    };
    let base = if index & 1 == 0 { 0 } else { 2 };
    let blocks = if index == 2 { 3 } else { 1 };
    let mut a = vec![0x7e00; base * 8 + rows * k];
    let mut b = vec![0x7e00; (base + k) * 8];
    let mut rng = 0xcafe_babeu32 ^ index;
    for value in a[base * 8..].iter_mut().chain(b[base * 8..].iter_mut()) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        *value = if index < 3 || index >= 64 {
            [0x3c00, 0xbc00, 0x3800, 0xb800][rng as usize & 3]
        } else {
            rng as u16
        };
    }
    if index == 3 {
        a[base * 8..].fill(0);
        b[base * 8..].fill(0);
        for row in 0..rows {
            a[base * 8 + row * k] = 0x3c00;
        }
        b[base * 8..base * 8 + 8]
            .copy_from_slice(&[0, 1, 0x3c00, 0x7c00, 0x7d00, 0xbc00, 0x7bff, 0x8001]);
    }
    let mut c = vec![0; rows * 8];
    let mut outputs = Vec::new();
    for _ in 0..blocks {
        for row in 0..rows {
            for inner in 0..k {
                for col in 0..8 {
                    let i = row * 8 + col;
                    c[i] = fp16::fma(
                        a[base * 8 + row * k + inner],
                        b[(base + inner) * 8 + col],
                        c[i],
                    );
                }
            }
        }
        outputs.extend_from_slice(&c);
    }
    CASE.with(|slot| {
        *slot.borrow_mut() = Some(Case {
            rows,
            k,
            base,
            blocks,
            a,
            b,
            c: outputs,
        })
    });
}

// Return rows, K, row base, and command count in successive 16-bit fields.
#[no_mangle]
pub extern "C" fn redmule_case_shape() -> u64 {
    CASE.with(|slot| {
        let slot = slot.borrow();
        let case = slot.as_ref().expect("case not loaded");
        case.rows as u64
            | (case.k as u64) << 16
            | (case.base as u64) << 32
            | (case.blocks as u64) << 48
    })
}

#[no_mangle]
pub extern "C" fn redmule_case_word(bank: u32, row: u32, half: u32) -> u64 {
    CASE.with(|slot| {
        let slot = slot.borrow();
        let case = slot.as_ref().expect("case not loaded");
        let data = match bank {
            0 => &case.a,
            1 => &case.b,
            2 => &case.c,
            _ => panic!("invalid case bank"),
        };
        let offset = row as usize * 8 + half as usize * 4;
        let mut word = 0;
        for lane in 0..4 {
            word |= (data[offset + lane] as u64) << (lane * 16);
        }
        word
    })
}
