use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;

#[derive(Clone, Copy)]
struct Pte {
    value: u64,
    error: bool,
}

struct Model {
    address_bits: u32,
    memory: HashMap<u64, Pte>,
    expected_reads: VecDeque<u64>,
}

#[derive(Default)]
struct Translation {
    address: u64,
    page_fault: bool,
    access_fault: bool,
    level: u32,
}

impl Translation {
    fn fault(page: bool, level: u32) -> Self {
        Self {
            page_fault: page,
            access_fault: !page,
            level,
            ..Self::default()
        }
    }
}

struct Request {
    address: u64,
    mode: u32,
    root: u64,
    privilege: u32,
    write: bool,
    execute: bool,
    sum: bool,
    mxr: bool,
}

impl Model {
    // Sequential architectural reference. Memory has only explicitly programmed
    // PTEs; Err(address) identifies an incomplete fixture, never an invented PTE.
    fn walk(&self, r: &Request, reads: &mut Vec<u64>) -> Result<Translation, u64> {
        let max_pa = (1u64 << self.address_bits) - 1;
        if r.privilege == 3 || r.mode == 0 {
            return Ok(if r.address > max_pa {
                Translation::fault(false, 0)
            } else {
                Translation { address: r.address, ..Translation::default() }
            });
        }
        if r.address > 0x0000_003f_ffff_ffff && r.address < 0xffff_ffc0_0000_0000 {
            return Ok(Translation::fault(true, 2));
        }
        let mut ppn = r.root;
        for level in (0..=2u32).rev() {
            let vpn = (r.address >> (12 + 9 * level)) & 511;
            let address = ppn * 4096 + vpn * 8;
            if address > max_pa {
                return Ok(Translation::fault(false, level));
            }
            reads.push(address);
            let pte = *self.memory.get(&address).ok_or(address)?;
            if pte.error {
                return Ok(Translation::fault(false, level));
            }
            let value = pte.value;
            let valid = value & 1 != 0;
            let readable = value & 2 != 0;
            let writable = value & 4 != 0;
            let executable = value & 8 != 0;
            if !valid || (writable && !readable) || value >> 54 != 0 {
                return Ok(Translation::fault(true, level));
            }
            ppn = (value >> 10) & ((1u64 << 44) - 1);
            if readable || executable {
                let user_page = value & 16 != 0;
                let privilege_ok = if r.privilege == 0 {
                    user_page
                } else {
                    !user_page || (r.sum && !r.execute)
                };
                let operation_ok = if r.execute {
                    executable
                } else if r.write {
                    writable
                } else {
                    readable || (r.mxr && executable)
                };
                let accessed = value & 64 != 0;
                let dirty = value & 128 != 0;
                let aligned = ppn & ((1u64 << (9 * level)) - 1) == 0;
                if !privilege_ok || !operation_ok || !accessed || (r.write && !dirty) || !aligned {
                    return Ok(Translation::fault(true, level));
                }
                let page_bytes = 4096u64 << (9 * level);
                let address = ppn * 4096 + (r.address & (page_bytes - 1));
                return Ok(if address > max_pa {
                    Translation::fault(false, level)
                } else {
                    Translation { address, level, ..Translation::default() }
                });
            }
            if level == 0 || value & 0xd0 != 0 {
                return Ok(Translation::fault(true, level));
            }
        }
        unreachable!("the bottom-level non-leaf faults before leaving the loop")
    }
}

#[no_mangle]
pub extern "C" fn mmu_ref_create(address_bits: u32) -> *mut c_void {
    assert!((44..=52).contains(&address_bits));
    Box::into_raw(Box::new(Model {
        address_bits,
        memory: HashMap::new(),
        expected_reads: VecDeque::new(),
    })).cast()
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<Model>()));
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_clear(model: *mut c_void) {
    let model = &mut *model.cast::<Model>();
    assert!(model.expected_reads.is_empty(), "fixture changed with unmatched PTE reads");
    model.memory.clear();
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_program(model: *mut c_void, address: u64, value: u64, error: u32) {
    assert_eq!(address % 8, 0);
    (*model.cast::<Model>()).memory.insert(address, Pte { value, error: error != 0 });
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_read(model: *mut c_void, address: u64, value: *mut u64, error: *mut u32) -> u32 {
    if let Some(pte) = (*model.cast::<Model>()).memory.get(&address) {
        *value = pte.value;
        *error = pte.error as u32;
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_pending(model: *mut c_void) -> u32 {
    (*model.cast::<Model>()).expected_reads.len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_peek(model: *mut c_void, address: *mut u64) -> u32 {
    if let Some(next) = (*model.cast::<Model>()).expected_reads.front() {
        *address = *next;
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_consume(model: *mut c_void, address: u64) -> u32 {
    let expected = &mut (*model.cast::<Model>()).expected_reads;
    if expected.front() != Some(&address) {
        return 0;
    }
    expected.pop_front();
    1
}

#[no_mangle]
pub unsafe extern "C" fn mmu_ref_translate(
    model: *mut c_void, address: u64, mode: u32, root: u64, privilege: u32,
    write: u32, execute: u32, sum: u32, mxr: u32,
    paddr: *mut u64, page_fault: *mut u32, access_fault: *mut u32, level: *mut u32,
) -> u32 {
    assert!(mode == 0 || mode == 8);
    assert!(matches!(privilege, 0 | 1 | 3));
    assert!(write == 0 || execute == 0);
    assert!(root < (1u64 << 44));
    let model = &mut *model.cast::<Model>();
    let request = Request { address, mode, root, privilege, write: write != 0, execute: execute != 0, sum: sum != 0, mxr: mxr != 0 };
    let mut reads = Vec::new();
    match model.walk(&request, &mut reads) {
        Ok(result) => {
            model.expected_reads.extend(reads);
            *paddr = result.address;
            *page_fault = result.page_fault as u32;
            *access_fault = result.access_fault as u32;
            *level = result.level;
            0
        }
        Err(address) => {
            *paddr = address;
            1
        }
    }
}
