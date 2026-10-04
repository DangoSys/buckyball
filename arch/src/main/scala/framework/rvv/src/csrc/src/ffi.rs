use crate::model::{Model, BANKS, BANK_BYTES};
use rvv::Memory;
use std::{
    path::Path,
    sync::{Mutex, OnceLock},
};
static MODEL: OnceLock<Mutex<Model>> = OnceLock::new();
fn model() -> std::sync::MutexGuard<'static, Model> {
    MODEL.get().unwrap().lock().unwrap()
}
#[no_mangle]
pub extern "C" fn rvv_model_init() {
    assert!(MODEL
        .set(Mutex::new(Model::new(
            &Path::new(env!("OUT_DIR")).join("kernels")
        )))
        .is_ok());
}
#[no_mangle]
pub extern "C" fn rvv_case_count() -> u32 {
    model().cases.len() as u32
}
#[no_mangle]
pub extern "C" fn rvv_case_select(n: u32) {
    model().select(n as usize);
}
#[no_mangle]
pub extern "C" fn rvv_case_meta(field: u32) -> u32 {
    let m = model();
    let c = &m.cases[m.current];
    match field {
        0 => m.current as u32 % 2,
        1 => c.entry,
        2 => c.program.len() as u32,
        3 => 0x80002000,
        4..=11 => c.args[(field - 4) as usize],
        _ => panic!("invalid case metadata field"),
    }
}
#[no_mangle]
pub extern "C" fn rvv_program_word(n: u32) -> u32 {
    let m = model();
    let c = &m.cases[m.current];
    let offset = n as usize * 4;
    u32::from_le_bytes(c.program[offset..offset + 4].try_into().unwrap())
}
#[no_mangle]
pub extern "C" fn rvv_expected_status(field: u32) -> u32 {
    let m = model();
    match field {
        0 => u32::from(m.fault.is_some()),
        1 => match m.fault {
            Some(fault) => fault.pc,
            None => m.cases[m.current].program.len() as u32,
        },
        2 => m.fault.map(|f| f.instruction).unwrap_or(0),
        3 => m.fault.map(|f| f.cause).unwrap_or(0),
        4 => m.fault.map(|f| f.value).unwrap_or(0),
        _ => panic!("invalid completion field"),
    }
}
#[no_mangle]
pub extern "C" fn rvv_memory_read(address: u32, size: u32) -> u64 {
    model().actual.read(address, 1usize << size).unwrap()
}
#[no_mangle]
pub extern "C" fn rvv_memory_error(address: u32, size: u32) -> u32 {
    u32::from(model().actual.read(address, 1usize << size).is_err())
}
#[no_mangle]
pub extern "C" fn rvv_memory_write(address: u32, size: u32, data: u64, mask: u32) {
    let mut m = model();
    let bytes = 1usize << size;
    assert_eq!(
        mask,
        (1u32 << bytes) - 1,
        "RVV requests must mask their exact access width"
    );
    m.actual.write(address, bytes, data).unwrap();
}
#[no_mangle]
pub extern "C" fn rvv_compare_memory() -> u32 {
    let m = model();
    for n in 0..BANK_BYTES * BANKS {
        if m.actual.0[n] != m.expected.0[n] {
            eprintln!(
                "{}: bank {} byte {} expected {:02x} got {:02x}",
                m.cases[m.current].name,
                n / BANK_BYTES,
                n % BANK_BYTES,
                m.expected.0[n],
                m.actual.0[n]
            );
            return 0;
        }
    }
    1
}

#[no_mangle]
pub extern "C" fn rvv_image_bytes() -> u32 {
    let m = model();
    24 + m.cases[m.current].program.len() as u32 + m.cases[m.current].constants.len() as u32
}
#[no_mangle]
pub extern "C" fn rvv_image_word(n: u32) -> u32 {
    let m = model();
    let c = &m.cases[m.current];
    let header = [
        0x31564b52,
        c.program.len() as u32,
        c.entry,
        0x80000000,
        c.constants.len() as u32,
        0,
    ];
    if n < 6 {
        return header[n as usize];
    }
    let offset = (n as usize - 6) * 4;
    if offset < c.program.len() {
        return u32::from_le_bytes(c.program[offset..offset + 4].try_into().unwrap());
    }
    let offset = offset - c.program.len();
    u32::from_le_bytes(c.constants[offset..offset + 4].try_into().unwrap())
}
#[no_mangle]
pub extern "C" fn rvv_memory_masked_write(address: u32, data: u64, mask: u32) {
    let mut m = model();
    for byte in 0..8 {
        if mask & (1 << byte) != 0 {
            m.actual
                .write(address + byte, 1, (data >> (byte * 8)) & 255)
                .unwrap();
        }
    }
}

#[no_mangle]
pub extern "C" fn rvv_compare_image() -> u32 {
    let m = model();
    let c = &m.cases[m.current];
    u32::from(m.actual.0.iter().enumerate().all(|(offset, value)| {
        if (2048..2096).contains(&offset) {
            *value == m.expected.0[offset]
        } else {
            *value == c.initial.0[offset]
        }
    }))
}
