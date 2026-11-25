use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
#[derive(Clone, Copy)]
struct Pte {
    value: u64,
    error: bool,
}
#[derive(Clone, Copy)]
struct Leaf {
    mode: u32,
    root: u64,
    vpn: u64,
    level: u32,
    pte: u64,
}
struct Model {
    address_bits: u32,
    memory: HashMap<u64, Pte>,
    expected_reads: VecDeque<u64>,
    tlb: Vec<Option<Leaf>>,
    victim: usize,
    epoch: u64,
    candidate: Option<(Leaf, u64)>,
    cached: bool,
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
fn canonical(address: u64) -> bool {
    address <= 0x0000_003f_ffff_ffff || address >= 0xffff_ffc0_0000_0000
}
impl Model {
    fn leaf(&self, r: &Request, level: u32, value: u64) -> Translation {
        let ppn = (value >> 10) & ((1u64 << 44) - 1);
        let user = value & 16 != 0;
        let privilege = if r.privilege == 0 {
            user
        } else {
            !user || (r.sum && !r.execute)
        };
        let operation = if r.execute {
            value & 8 != 0
        } else if r.write {
            value & 4 != 0
        } else {
            value & 2 != 0 || (r.mxr && value & 8 != 0)
        };
        if !privilege
            || !operation
            || value & 64 == 0
            || (r.write && value & 128 == 0)
            || ppn & ((1u64 << (9 * level)) - 1) != 0
        {
            return Translation::fault(true, level);
        }
        let bytes = 4096u64 << (9 * level);
        let address = ppn * 4096 + (r.address & (bytes - 1));
        if address >= (1u64 << self.address_bits) {
            Translation::fault(false, level)
        } else {
            Translation {
                address,
                level,
                ..Translation::default()
            }
        }
    }
    fn cached(&self, r: &Request) -> Option<Translation> {
        if r.privilege == 3 || r.mode == 0 || !canonical(r.address) {
            return None;
        }
        self.tlb
            .iter()
            .flatten()
            .find(|entry| {
                entry.mode == r.mode
                    && entry.root == r.root
                    && entry.vpn == r.address >> (12 + 9 * entry.level)
            })
            .map(|entry| self.leaf(r, entry.level, entry.pte))
    }
    fn walk(&self, r: &Request, reads: &mut Vec<u64>) -> Result<Translation, u64> {
        let limit = 1u64 << self.address_bits;
        if r.privilege == 3 || r.mode == 0 {
            return Ok(if r.address >= limit {
                Translation::fault(false, 0)
            } else {
                Translation {
                    address: r.address,
                    ..Translation::default()
                }
            });
        }
        if !canonical(r.address) {
            return Ok(Translation::fault(true, 2));
        }
        let mut ppn = r.root;
        for level in (0..=2u32).rev() {
            let address = ppn * 4096 + ((r.address >> (12 + 9 * level)) & 511) * 8;
            if address >= limit {
                return Ok(Translation::fault(false, level));
            }
            reads.push(address);
            let pte = *self.memory.get(&address).ok_or(address)?;
            if pte.error {
                return Ok(Translation::fault(false, level));
            }
            let value = pte.value;
            if value & 1 == 0 || value & 6 == 4 || value >> 54 != 0 {
                return Ok(Translation::fault(true, level));
            }
            ppn = (value >> 10) & ((1u64 << 44) - 1);
            if value & 10 != 0 {
                return Ok(self.leaf(r, level, value));
            }
            if level == 0 || value & 0xd0 != 0 {
                return Ok(Translation::fault(true, level));
            }
        }
        unreachable!()
    }
}
#[no_mangle]
pub extern "C" fn mmu_ref_create(address_bits: u32, entries: u32) -> *mut c_void {
    assert!((44..=52).contains(&address_bits));
    Box::into_raw(Box::new(Model {
        address_bits,
        memory: HashMap::new(),
        expected_reads: VecDeque::new(),
        tlb: vec![None; entries as usize],
        victim: 0,
        epoch: 0,
        candidate: None,
        cached: false,
    }))
    .cast()
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_destroy(p: *mut c_void) {
    drop(Box::from_raw(p.cast::<Model>()));
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_clear(p: *mut c_void) {
    let m = &mut *p.cast::<Model>();
    assert!(m.expected_reads.is_empty());
    m.memory.clear();
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_program(p: *mut c_void, address: u64, value: u64, error: u32) {
    assert_eq!(address % 8, 0);
    (*p.cast::<Model>()).memory.insert(
        address,
        Pte {
            value,
            error: error != 0,
        },
    );
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_read(
    p: *mut c_void,
    address: u64,
    value: *mut u64,
    error: *mut u32,
) -> u32 {
    if let Some(pte) = (*p.cast::<Model>()).memory.get(&address) {
        *value = pte.value;
        *error = u32::from(pte.error);
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_pending(p: *mut c_void) -> u32 {
    (*p.cast::<Model>()).expected_reads.len() as u32
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_peek(p: *mut c_void, address: *mut u64) -> u32 {
    if let Some(next) = (*p.cast::<Model>()).expected_reads.front() {
        *address = *next;
        1
    } else {
        0
    }
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_consume(p: *mut c_void, address: u64) -> u32 {
    let queue = &mut (*p.cast::<Model>()).expected_reads;
    if queue.front() != Some(&address) {
        return 0;
    }
    queue.pop_front();
    1
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_translate(
    p: *mut c_void,
    address: u64,
    mode: u32,
    root: u64,
    privilege: u32,
    write: u32,
    execute: u32,
    sum: u32,
    mxr: u32,
    paddr: *mut u64,
    page_fault: *mut u32,
    access_fault: *mut u32,
    level: *mut u32,
) -> u32 {
    assert!(mode == 0 || mode == 8);
    assert!(matches!(privilege, 0 | 1 | 3));
    assert!(write == 0 || execute == 0);
    assert!(root < 1u64 << 44);
    let m = &mut *p.cast::<Model>();
    let r = Request {
        address,
        mode,
        root,
        privilege,
        write: write != 0,
        execute: execute != 0,
        sum: sum != 0,
        mxr: mxr != 0,
    };
    let mut reads = Vec::new();
    m.candidate = None;
    let cached = m.cached(&r);
    m.cached = cached.is_some();
    let result = match cached {
        Some(answer) => Ok(answer),
        None => m.walk(&r, &mut reads),
    };
    match result {
        Ok(answer) => {
            if !answer.page_fault && !answer.access_fault && !reads.is_empty() {
                let pte = m.memory[reads.last().unwrap()].value;
                m.candidate = Some((
                    Leaf {
                        mode,
                        root,
                        vpn: address >> (12 + 9 * answer.level),
                        level: answer.level,
                        pte,
                    },
                    m.epoch,
                ));
            }
            m.expected_reads.extend(reads);
            *paddr = answer.address;
            *page_fault = u32::from(answer.page_fault);
            *access_fault = u32::from(answer.access_fault);
            *level = answer.level;
            0
        }
        Err(address) => {
            *paddr = address;
            1
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_cached(p: *mut c_void) -> u32 {
    u32::from((*p.cast::<Model>()).cached)
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_result(p: *mut c_void) {
    let m = &mut *p.cast::<Model>();
    if m.expected_reads.is_empty() {
        if let Some((leaf, epoch)) = m.candidate.take() {
            if epoch == m.epoch && !m.tlb.is_empty() {
                m.tlb[m.victim] = Some(leaf);
                m.victim = (m.victim + 1) % m.tlb.len();
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_flush(p: *mut c_void) {
    let m = &mut *p.cast::<Model>();
    m.tlb.fill(None);
    m.epoch += 1;
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_reset(p: *mut c_void) {
    let m = &mut *p.cast::<Model>();
    m.tlb.fill(None);
    m.epoch += 1;
    m.victim = 0;
    m.candidate = None;
    m.expected_reads.clear();
}
#[no_mangle]
pub unsafe extern "C" fn mmu_ref_lookup(
    p: *mut c_void,
    address: u64,
    mode: u32,
    root: u64,
    privilege: u32,
    write: u32,
    execute: u32,
    sum: u32,
    mxr: u32,
    paddr: *mut u64,
) -> u32 {
    let m = &*p.cast::<Model>();
    let r = Request {
        address,
        mode,
        root,
        privilege,
        write: write != 0,
        execute: execute != 0,
        sum: sum != 0,
        mxr: mxr != 0,
    };
    if privilege == 3 || mode == 0 {
        *paddr = address;
        return u32::from(address < 1u64 << m.address_bits);
    }
    if let Some(answer) = m.cached(&r) {
        *paddr = answer.address;
        u32::from(!answer.page_fault && !answer.access_fault)
    } else {
        0
    }
}
