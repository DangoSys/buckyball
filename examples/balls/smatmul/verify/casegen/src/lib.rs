mod casegen;
mod model;

use std::cell::RefCell;

use casegen::{MatrixCase, MatrixCmd, MAX_WORDS};

thread_local! {
    static CURRENT: RefCell<Option<MatrixCase>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn smatmul_case_load(seed: u32, index: u32, bid: u32, out_bw: u32) -> i32 {
    let case = casegen::gen_case(seed, index, bid, out_bw as usize);
    let na = case.cmd.num_a_words as usize;
    let nb = case.cmd.num_b_words as usize;
    if na > MAX_WORDS || nb > MAX_WORDS {
        panic!("smatmul_case_load: word count out of range a={na} b={nb}");
    }
    CURRENT.with(|c| *c.borrow_mut() = Some(case));
    0
}

fn current_case<F: FnOnce(&MatrixCase) -> R, R>(f: F) -> R {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(case) => f(case),
        None => panic!("matrix_case: no case loaded; call smatmul_case_load first"),
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_cmd(out_ptr: *mut MatrixCmd) {
    current_case(|case| unsafe {
        if out_ptr.is_null() {
            panic!("smatmul_case_cmd: null out_ptr");
        }
        *out_ptr = case.cmd;
    });
}

#[no_mangle]
pub extern "C" fn smatmul_case_a_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("smatmul_case_a_word_lo: word_index out of range");
    }
    current_case(|case| case.a_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn smatmul_case_a_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("smatmul_case_a_word_hi: word_index out of range");
    }
    current_case(|case| case.a_word_hi(word_index as usize))
}

#[no_mangle]
pub extern "C" fn smatmul_case_b_word_lo(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("smatmul_case_b_word_lo: word_index out of range");
    }
    current_case(|case| case.b_word_lo(word_index as usize))
}

#[no_mangle]
pub extern "C" fn smatmul_case_b_word_hi(word_index: u32) -> u64 {
    if word_index as usize >= MAX_WORDS {
        panic!("smatmul_case_b_word_hi: word_index out of range");
    }
    current_case(|case| case.b_word_hi(word_index as usize))
}

#[no_mangle]
pub extern "C" fn smatmul_case_num_writes() -> u32 {
    current_case(|case| case.cmd.num_writes)
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_group(i: u32) -> u32 {
    current_case(|case| {
        let idx = i as usize;
        if idx >= case.writes.len() {
            panic!("smatmul_case_write_group: index out of range");
        }
        case.writes[idx].group
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_addr(i: u32) -> u32 {
    current_case(|case| {
        let idx = i as usize;
        if idx >= case.writes.len() {
            panic!("smatmul_case_write_addr: index out of range");
        }
        case.writes[idx].addr
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_data_lo(i: u32) -> u64 {
    current_case(|case| {
        let idx = i as usize;
        if idx >= case.writes.len() {
            panic!("smatmul_case_write_data_lo: index out of range");
        }
        u64::from_le_bytes(case.writes[idx].data[0..8].try_into().unwrap())
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_data_hi(i: u32) -> u64 {
    current_case(|case| {
        let idx = i as usize;
        if idx >= case.writes.len() {
            panic!("smatmul_case_write_data_hi: index out of range");
        }
        u64::from_le_bytes(case.writes[idx].data[8..16].try_into().unwrap())
    })
}

#[no_mangle]
pub extern "C" fn smatmul_case_write_mask(i: u32) -> u32 {
    current_case(|case| {
        let idx = i as usize;
        if idx >= case.writes.len() {
            panic!("smatmul_case_write_mask: index out of range");
        }
        case.writes[idx].mask as u32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_load_then_cmd_and_words() {
        assert_eq!(smatmul_case_load(0x1234, 0, 1, 2), 0);
        let mut cmd = MatrixCmd {
            bid: 0,
            ws: 0,
            m: 0,
            n: 0,
            k: 0,
            op1_bank: 0,
            op2_bank: 0,
            wr_bank: 0,
            rob_id: 0,
            rs1_lo: 0,
            rs1_hi: 0,
            rs2_lo: 0,
            rs2_hi: 0,
            num_a_words: 0,
            num_b_words: 0,
            num_writes: 0,
        };
        smatmul_case_cmd(&mut cmd as *mut MatrixCmd);
        assert_eq!(cmd.ws, 0);
        assert_eq!(cmd.num_writes, 4);
        let _ = smatmul_case_a_word_lo(0);
        let _ = smatmul_case_b_word_lo(0);
        assert_eq!(smatmul_case_write_group(0), 0);
    }

    #[test]
    #[should_panic(expected = "no case loaded")]
    fn current_case_panics_without_load() {
        CURRENT.with(|c| *c.borrow_mut() = None);
        current_case(|_| ());
    }
}
