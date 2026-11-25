// Reference Ant hart: RV64IM in machine mode with code, TLS and TSS as its only memory,
// `csrr mcycle` as its only CSR, raw custom-0x7b commands and ECALL as task exit.
// Every step reports the retirement the RTL trace must show for the same instruction.

use crate::arithmetic::ant_arithmetic;

struct Hart {
    code: Vec<u8>,
    tls_base: u64,
    tls: Vec<u8>,
    tss_base: u64,
    tss: Vec<u8>,
    registers: [u64; 32],
    pc: u64,
}

impl Hart {
    fn memory(&mut self, address: u64, bytes: u64) -> &mut [u8] {
        assert!(address % bytes == 0, "misaligned access 0x{address:x}");
        for (base, store) in [(self.tls_base, &mut self.tls), (self.tss_base, &mut self.tss)] {
            if address >= base && address + bytes <= base + store.len() as u64 {
                let offset = (address - base) as usize;
                return &mut store[offset..offset + bytes as usize];
            }
        }
        panic!("access outside TLS/TSS: 0x{address:x}")
    }
}

fn sext(value: u64, bits: u32) -> u64 {
    ((value << (64 - bits)) as i64 >> (64 - bits)) as u64
}

#[no_mangle]
pub extern "C" fn ant_hart_create(
    code_bytes: u32,
    tls_base: u64,
    tls_bytes: u32,
    tss_base: u64,
    tss_bytes: u32,
) -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(Hart {
        code: vec![0; code_bytes as usize],
        tls_base,
        tls: vec![0; tls_bytes as usize],
        tss_base,
        tss: vec![0; tss_bytes as usize],
        registers: [0; 32],
        pc: 0,
    }))
    .cast()
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_destroy(handle: *mut std::ffi::c_void) {
    drop(unsafe { Box::from_raw(handle.cast::<Hart>()) });
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_code(handle: *mut std::ffi::c_void, offset: u32, word: u32) {
    let h = unsafe { &mut *handle.cast::<Hart>() };
    h.code[offset as usize..offset as usize + 4].copy_from_slice(&word.to_le_bytes());
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_store(handle: *mut std::ffi::c_void, address: u64, value: u64) {
    let h = unsafe { &mut *handle.cast::<Hart>() };
    h.memory(address, 8).copy_from_slice(&value.to_le_bytes());
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_load(handle: *mut std::ffi::c_void, address: u64) -> u64 {
    let h = unsafe { &mut *handle.cast::<Hart>() };
    u64::from_le_bytes(h.memory(address, 8).try_into().unwrap())
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_register(handle: *mut std::ffi::c_void, index: u32) -> u64 {
    unsafe { &*handle.cast::<Hart>() }.registers[index as usize]
}

#[no_mangle]
pub unsafe extern "C" fn ant_hart_start(
    handle: *mut std::ffi::c_void,
    entry: u64,
    argument: u64,
    stack: u64,
) {
    let h = unsafe { &mut *handle.cast::<Hart>() };
    h.registers = [0; 32];
    h.registers[2] = stack;
    h.registers[4] = h.tls_base;
    h.registers[10] = argument;
    h.pc = entry;
}

// Executes one instruction. Returns 0 for a retirement, 1 for ECALL (data = a0, no
// retirement) and 2 for a retired custom command. `cycle` supplies the mcycle value the
// RTL observed, the only state the reference cannot predict.
#[no_mangle]
pub unsafe extern "C" fn ant_hart_step(
    handle: *mut std::ffi::c_void,
    cycle: u64,
    pc: *mut u64,
    instruction: *mut u32,
    rd: *mut u32,
    data: *mut u64,
) -> u32 {
    let h = unsafe { &mut *handle.cast::<Hart>() };
    let offset = h.pc as usize;
    let i = u32::from_le_bytes(h.code[offset..offset + 4].try_into().unwrap());
    let x = h.registers;
    let opcode = i & 0x7f;
    let f3 = (i >> 12) & 7;
    let f7 = i >> 25;
    let d = (i >> 7) & 31;
    let a = x[((i >> 15) & 31) as usize];
    let b = x[((i >> 20) & 31) as usize];
    let word = opcode & 8 != 0;
    let imm_i = sext((i >> 20) as u64, 12);
    let imm_s = sext((((i >> 25) << 5) | ((i >> 7) & 31)) as u64, 12);
    let imm_b = sext(
        ((((i >> 31) & 1) << 12) | (((i >> 7) & 1) << 11) | (((i >> 25) & 63) << 5) | (((i >> 8) & 15) << 1))
            as u64,
        13,
    );
    let imm_u = sext((i & 0xffff_f000) as u64, 32);
    let imm_j = sext(
        ((((i >> 31) & 1) << 20) | (((i >> 12) & 255) << 12) | (((i >> 20) & 1) << 11) | (((i >> 21) & 1023) << 1))
            as u64,
        21,
    );
    let next = h.pc.wrapping_add(4);
    let fix = |v: u64| if word { sext(v & 0xffff_ffff, 32) } else { v };
    let (target, destination, value, kind) = match opcode {
        0x37 => (next, d, imm_u, 0),
        0x17 => (next, d, h.pc.wrapping_add(imm_u), 0),
        0x6f => (h.pc.wrapping_add(imm_j), d, next, 0),
        0x67 => (a.wrapping_add(imm_i) & !1, d, next, 0),
        0x63 => {
            let taken = match f3 {
                0 => a == b,
                1 => a != b,
                4 => (a as i64) < (b as i64),
                5 => (a as i64) >= (b as i64),
                6 => a < b,
                7 => a >= b,
                _ => panic!("illegal branch 0x{i:08x}"),
            };
            (if taken { h.pc.wrapping_add(imm_b) } else { next }, 0, next, 0)
        }
        0x03 => {
            let bytes = 1u64 << (f3 & 3);
            let raw = h.memory(a.wrapping_add(imm_i), bytes);
            let mut v = 0u64;
            for (n, byte) in raw.iter().enumerate() {
                v |= (*byte as u64) << (8 * n);
            }
            let v = if f3 < 4 && bytes < 8 { sext(v, 8 * bytes as u32) } else { v };
            (next, d, v, 0)
        }
        0x23 => {
            let bytes = 1u64 << (f3 & 3);
            let raw = h.memory(a.wrapping_add(imm_s), bytes);
            for (n, byte) in raw.iter_mut().enumerate() {
                *byte = (b >> (8 * n)) as u8;
            }
            (next, 0, 0, 0)
        }
        0x13 | 0x1b | 0x33 | 0x3b if !(opcode & 0x20 != 0 && f7 == 1) => {
            let rhs = if opcode & 0x20 != 0 { b } else { imm_i };
            let shift = (rhs & if word { 31 } else { 63 }) as u32;
            let sub = opcode & 0x20 != 0 && f7 == 32;
            let v = match f3 {
                0 => if sub { a.wrapping_sub(rhs) } else { a.wrapping_add(rhs) },
                1 => a << shift,
                2 => ((a as i64) < (rhs as i64)) as u64,
                3 => (a < rhs) as u64,
                4 => a ^ rhs,
                5 if i & (1 << 30) != 0 => {
                    if word { (a as u32 as i32 >> shift) as i64 as u64 } else { ((a as i64) >> shift) as u64 }
                }
                5 => if word { (a as u32 >> shift) as u64 } else { a >> shift },
                6 => a | rhs,
                _ => a & rhs,
            };
            (next, d, fix(v), 0)
        }
        0x33 | 0x3b => (next, d, ant_arithmetic(a, b, f3, word), 0),
        0x7b => (next, 0, 0, 2),
        0x0f => (next, 0, 0, 0),
        0x73 if i == 0x73 => (next, 10, x[10], 1),
        0x73 => {
            assert_eq!(i & 0xffff_f07f, 0xb000_2073, "unsupported system instruction 0x{i:08x}");
            (next, d, cycle, 0)
        }
        _ => panic!("illegal instruction 0x{i:08x}"),
    };
    unsafe {
        *pc = h.pc;
        *instruction = i;
        *rd = destination;
        *data = if destination == 0 { 0 } else { value };
    }
    if destination != 0 && kind != 1 {
        h.registers[destination as usize] = value;
    }
    if kind == 1 {
        unsafe { *data = x[10] };
    }
    h.pc = target;
    kind
}
