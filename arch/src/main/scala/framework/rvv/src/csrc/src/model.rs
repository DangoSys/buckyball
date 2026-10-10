include!(concat!(env!("OUT_DIR"), "/params.rs"));

use rvv::{Engine, Fault, Memory, MemoryError};
use std::path::Path;

pub const BANK_BYTES: usize = 4096;
pub const BANKS: usize = 6;
#[derive(Clone)]
pub struct Banks(pub Vec<u8>);
impl Banks {
    fn range(&self, address: u64, bytes: usize) -> Result<std::ops::Range<usize>, MemoryError> {
        let bank = (address >> 32) as usize;
        let offset = (address & 0xffff_ffff) as usize;
        if bank >= BANKS || offset + bytes > BANK_BYTES {
            return Err(MemoryError);
        }
        Ok(bank * BANK_BYTES + offset..bank * BANK_BYTES + offset + bytes)
    }
    fn put(&mut self, bank: usize, offset: usize, data: &[u8]) {
        self.0[bank * BANK_BYTES + offset..bank * BANK_BYTES + offset + data.len()]
            .copy_from_slice(data);
    }
}
impl Memory for Banks {
    fn ball_command(&mut self, _: u32, _: u64, _: u64) -> Result<u64, MemoryError> {
        Err(MemoryError)
    }
    fn read(&mut self, address: u64, bytes: usize) -> Result<u64, MemoryError> {
        let mut value = [0u8; 8];
        value[..bytes].copy_from_slice(&self.0[self.range(address, bytes)?]);
        Ok(u64::from_le_bytes(value))
    }
    fn write(&mut self, address: u64, bytes: usize, value: u64) -> Result<(), MemoryError> {
        let range = self.range(address, bytes)?;
        self.0[range].copy_from_slice(&value.to_le_bytes()[..bytes]);
        Ok(())
    }
}
pub struct Case {
    pub name: String,
    pub program: Vec<u8>,
    pub constants: Vec<u8>,
    pub entry: u32,
    pub args: [u64; 8],
    pub initial: Banks,
    pub expected_cause: u32,
    pub fixed_expected: Option<Banks>,
}
pub struct Model {
    pub cases: Vec<Case>,
    pub current: usize,
    pub actual: Banks,
    pub expected: Banks,
    pub engine: Engine,
    pub fault: Option<Fault>,
}
fn i(op: u32, rd: u32, f: u32, rs: u32, imm: i32) -> u32 {
    ((imm as u32 & 4095) << 20) | rs << 15 | f << 12 | rd << 7 | op
}
fn s(op: u32, f: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
    (imm >> 5) << 25 | rs2 << 20 | rs1 << 15 | f << 12 | (imm & 31) << 7 | op
}
fn r(op: u32, rd: u32, f: u32, rs1: u32, rs2: u32, f7: u32) -> u32 {
    f7 << 25 | rs2 << 20 | rs1 << 15 | f << 12 | rd << 7 | op
}
fn initial() -> Banks {
    let mut b = Banks(vec![0; BANKS * BANK_BYTES]);
    for offset in 0..BANK_BYTES {
        b.0[BANK_BYTES + offset] = (offset as u8).wrapping_mul(73).wrapping_add(19);
    }
    for n in 0..256 {
        let x = (n as f32 - 123.0) / 31.0;
        b.put(2, n * 4, &x.to_le_bytes());
        b.put(3, n * 4, &(0.25 + n as f32 / 128.0).to_le_bytes());
    }
    b
}
fn micro(name: &str, words: &[u32], args: [u64; 8]) -> Case {
    Case {
        name: name.into(),
        program: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        constants: vec![0; 512],
        entry: 0,
        args,
        initial: initial(),
        expected_cause: 0,
        fixed_expected: None,
    }
}
fn elf(path: &Path, name: &str, args: [u64; 8]) -> Case {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..7], b"\x7fELF\x02\x01\x01");
    let u64_at = |p| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    let u32_at = |p| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
    let u16_at = |p| u16::from_le_bytes(bytes[p..p + 2].try_into().unwrap());
    assert_eq!(u16_at(18), 243);
    let mut c = micro(name, &[], args);
    c.entry = u32::try_from(u64_at(24)).expect("kernel entry exceeds local address width");
    assert_eq!(u16_at(58), 64, "invalid ELF64 section header size");
    let sections = usize::try_from(u64_at(40)).unwrap();
    for n in 0..u16_at(60) as usize {
        let p = sections.checked_add(n.checked_mul(64).unwrap()).unwrap();
        let flags = u64_at(p + 8);
        let address = u32::try_from(u64_at(p + 16)).expect("kernel section exceeds local address width");
        let size = usize::try_from(u64_at(p + 32)).unwrap();
        if flags & 2 == 0 || size == 0 {
            continue;
        }
        let offset = usize::try_from(u64_at(p + 24)).unwrap();
        if flags & 4 != 0 {
            assert_eq!(address, 0);
            c.program = bytes[offset..offset.checked_add(size).unwrap()].to_vec();
        } else {
            assert_ne!(
                u32_at(p + 4),
                8,
                "allocated NOBITS requires explicit initialization"
            );
            assert_eq!(address, 0x40000000);
            assert_eq!(flags & 1, 0, "kernel globals must be read-only");
            c.constants.resize(size.max(512), 0);
            c.constants[..size].copy_from_slice(&bytes[offset..offset.checked_add(size).unwrap()]);
        }
    }
    assert!(!c.program.is_empty() && c.program.len() <= 4096 && c.program.len() % 4 == 0);
    c
}
// Independent integer arithmetic gold: the BEMU Engine does not implement these fixed-point opcodes.
fn fixed_gold(case: &Case, bits: usize, op: u32, mode: u32) -> Banks {
    let mut gold = case.initial.clone();
    let bytes = bits / 8;
    let mask = (1i128 << bits) - 1;
    let minimum = -(1i128 << (bits - 1));
    let maximum = (1i128 << (bits - 1)) - 1;
    let read = |offset: usize, size: usize| {
        let mut value = [0; 8];
        value[..size].copy_from_slice(&case.initial.0[2 * BANK_BYTES + offset..2 * BANK_BYTES + offset + size]);
        u64::from_le_bytes(value) as i128
    };
    let signed = |value: i128, width: usize| if value & (1i128 << (width - 1)) != 0 { value - (1i128 << width) } else { value };
    let mut saturated = false;
    for lane in 0..5 {
        let a = read(lane * bytes, bytes);
        let b = read(128 + lane * bytes, bytes);
        let mut result = a;
        if mode != 3 || lane >= 2 {
            let (value, shift) = match op {
                8 => (a + b, 1), 9 => (signed(a, bits) + signed(b, bits), 1),
                10 => (a - b, 1), 11 => (signed(a, bits) - signed(b, bits), 1),
                39 => (signed(a, bits) * signed(b, bits), bits - 1),
                42 => (a, b as usize & (bits - 1)),
                43 => (signed(a, bits), b as usize & (bits - 1)),
                46 => (read(lane * bytes * 2, bytes * 2), b as usize & (bits * 2 - 1)),
                47 => (signed(read(lane * bytes * 2, bytes * 2), bits * 2), b as usize & (bits * 2 - 1)),
                _ => unreachable!(),
            };
            let truncated = value >> shift;
            let half = shift != 0 && value & (1i128 << (shift - 1)) != 0;
            let low = shift != 0 && value & ((1i128 << (shift - 1)) - 1) != 0;
            let increment = match mode {
                0 => half, 1 => half && (low || truncated & 1 != 0), 2 => false,
                3 => truncated & 1 == 0 && value & ((1i128 << shift) - 1) != 0,
                _ => unreachable!(),
            };
            result = truncated + i128::from(increment);
            if op == 39 || op == 47 {
                saturated |= result < minimum || result > maximum;
                result = result.clamp(minimum, maximum);
            } else if op == 46 {
                saturated |= result > mask;
                result = result.min(mask);
            }
        }
        gold.put(1, lane * bytes, &((result & mask) as u64).to_le_bytes()[..bytes]);
    }
    gold.put(1, 128, &u32::from(saturated).to_le_bytes());
    gold
}

// Additional coverage for balanced integer reductions and bounded rounding carry.
fn integer_depth_cases(args: [u64; 8]) -> Vec<Case> {
    let mut cases = vec![];
    for sew in 0..3 {
        let bits = 8usize << sew;
        let mask = u64::MAX >> (64 - bits);
        let values = [1u64 << (bits - 1), mask >> 1, 0, 1, mask];
        let width = [0, 5, 6, 7][sew as usize];
        for reduction in [true, false] {
            let operations: Vec<(u32, u32)> = if reduction {
                (0..8).map(|op| (op, 2)).collect()
            } else {
                let mut ops = vec![(8, 2), (9, 2), (10, 2), (11, 2), (39, 0), (42, 0), (43, 0)];
                if sew < 2 { ops.extend([(46, 0), (47, 0)]); }
                ops
            };
            for (op, kind) in operations {
                for mode in 0..(if reduction { 5 } else { 4 }) {
                    let mut a = args;
                    a[2] = 5;
                    let lmul = if 5 * bits > VLEN { 1 } else { 0 };
                    let mut words = vec![
                        i(0x57, 5, 7, 12, ((sew << 3) | lmul) as i32),
                        (1 << 25) | (11 << 15) | (width << 12) | (8 << 7) | 7,
                        i(0x13, 6, 0, 11, 128),
                        (1 << 25) | (6 << 15) | (width << 12) | (16 << 7) | 7,
                        (1 << 25) | (11 << 15) | (width << 12) | (24 << 7) | 7,
                    ];
                    let mut vm = 1;
                    if reduction {
                        if mode == 1 {
                            words.push((0x19 << 26) | (1 << 25) | (8 << 20) | (3 << 12) | 0x57);
                            vm = 0;
                        } else if mode == 2 {
                            words.push((0x1b << 26) | (1 << 25) | (2 << 12) | 0x57);
                            vm = 0;
                        } else if mode == 3 {
                            words.push(i(0x57, 5, 7, 0, 0xc00 | (sew << 3) as i32));
                        } else if mode == 4 {
                            words.push(i(0x73, 0, 5, 1, 8));
                        }
                    } else {
                        words.push(i(0x73, 0, 5, mode, 0xa));
                        words.push(i(0x73, 0, 5, 0, 9));
                        if op == 46 || op == 47 {
                            words.push((1 << 25) | (11 << 15) | ([5, 6, 7][sew as usize] << 12) | (8 << 7) | 7);
                        }
                    }
                    if !reduction && mode == 3 { words.push(i(0x73, 0, 5, 2, 8)); }
                    words.push((op << 26) | (vm << 25) | (8 << 20) | (16 << 15) | (kind << 12) | (24 << 7) | 0x57);
                    words.push(i(0x57, 5, 7, 12, ((sew << 3) | lmul) as i32));
                    words.push((1 << 25) | (10 << 15) | (width << 12) | (24 << 7) | 0x27);
                    if !reduction {
                        words.push(i(0x73, 6, 2, 0, 9));
                        words.push(s(0x23, 2, 10, 6, 128));
                    }
                    words.push(0x8067);
                    let mut case = micro(&format!("depth_{}_e{bits}_op{op}_mode{mode}", if reduction {"reduce"} else {"round"}), &words, a);
                    for (index, value) in values.into_iter().enumerate() {
                        let bytes = bits / 8;
                        let rhs = if reduction || op == 39 || kind == 2 { value } else { [0, 1, bits as u64 - 1, bits as u64 / 2, 3][index] };
                        case.initial.put(2, index * bytes, &value.to_le_bytes()[..bytes]);
                        case.initial.put(2, 128 + index * bytes, &rhs.to_le_bytes()[..bytes]);
                    }
                    if !reduction { case.fixed_expected = Some(fixed_gold(&case, bits, op, mode)); }
                    if reduction && mode == 4 { case.expected_cause = 2; }
                    cases.push(case);
                }
            }
        }
    }
    cases
}

impl Model {
    pub fn new(kernel_dir: &Path) -> Self {
        let mut cases = vec![];
        let mut args = [0; 8];
        args[0] = 1 << 32;
        args[1] = 2 << 32;
        for kernel in ["silu", "swiglu", "snake", "quant"] {
            let lengths: &[u32] = if kernel == "quant" {
                &[32, 64]
            } else {
                &[1, 31, 33, 65]
            };
            for &n in lengths {
                let mut a = args;
                match kernel {
                    "swiglu" => {
                        a[2] = 3 << 32;
                        a[3] = u64::from(n);
                    }
                    "snake" => {
                        a[2] = 2;
                        a[3] = u64::from(n);
                        a[4] = 48;
                        a[5] = 56;
                    }
                    "quant" => {
                        a[1] = args[0] + 1024;
                        a[2] = args[1];
                        a[3] = u64::from(n);
                        a[4] = args[0] + 1536;
                    }
                    _ => a[2] = u64::from(n),
                }
                if kernel == "silu" { a[3] = args[0] + 1024; }
                if kernel == "swiglu" { a[4] = args[0] + 1024; }
                let mut c = elf(
                    &kernel_dir.join(format!("{kernel}.elf")),
                    &format!("{kernel}_{n}"),
                    a,
                );
                if kernel == "snake" {
                    c.initial.put(
                        0,
                        48,
                        &[
                            0.0f32.to_le_bytes(),
                            0.5f32.to_le_bytes(),
                            0.25f32.to_le_bytes(),
                            (-0.25f32).to_le_bytes(),
                        ]
                        .concat(),
                    );
                }
                cases.push(c);
            }
        }
        // Unaligned loads/stores and every RV32M operation, including divide by zero and overflow.
        for &(a, b) in &[
            (17u32, 3u32),
            (0x80000000, 0xffffffff),
            (0xffffffef, 0),
            (0xffffffff, 7),
        ] {
            let mut aargs = args;
            aargs[2] = u64::from(a);
            aargs[3] = u64::from(b);
            let mut words = vec![];
            for f in 0..8 {
                words.push(r(0x33, 5, f, 12, 13, 1));
                words.push(s(0x23, 2, 10, 5, f * 4 + 1));
            }
            for f in [0, 1, 2, 4, 5] {
                words.push(i(3, 6, f, 10, 1));
                words.push(s(0x23, 2, 10, 6, 48 + f * 4));
            }
            words.push(0x8067);
            cases.push(micro(&format!("scalar_m_{a:x}_{b:x}"), &words, aargs));
        }
        for funct in [0, 1, 4, 5, 6, 7] {
            for &(left, right) in &[(0xffffffff, 1), (7, 7)] {
                let mut a = args;
                a[2] = left;
                a[3] = right;
                let words = [
                    i(0x13, 5, 0, 0, 0),
                    (13 << 20) | (12 << 15) | (funct << 12) | (4 << 8) | 0x63,
                    i(0x13, 5, 0, 0, 1),
                    s(0x23, 2, 10, 5, 0),
                    0x8067,
                ];
                cases.push(micro(
                    &format!("branch_{funct}_{left:x}_{right:x}"),
                    &words,
                    a,
                ));
            }
        }
        for sew in [2, 3] {
            for (vl, start) in [(0, 0), (4, 0), (4, 1), (4, 4)] {
                let width = if sew == 2 { 6 } else { 7 };
                let words = [
                    i(0x57, 5, 7, 4, 0xc00 | (sew << 3)),
                    (1 << 25) | (11 << 15) | (width << 12) | (9 << 7) | 7,
                    i(0x57, 5, 7, 12, (sew << 3) | 3),
                    i(0x73, 0, 5, start, 8),
                    r(0x57, 9, 6, 13, 0, 33),
                    i(0x57, 5, 7, 4, 0xc00 | (sew << 3)),
                    (1 << 25) | (10 << 15) | (width << 12) | (9 << 7) | 0x27,
                    i(0x73, 7, 2, 0, 1),
                    s(0x23, 2, 10, 7, 64),
                    0x8067,
                ];
                let mut a = args;
                a[2] = vl;
                a[3] = 0x80000001;
                let mut c = micro(&format!("scalar_insert_e{}_vl{vl}_start{start}", 8 << sew), &words, a);
                if sew == 3 && ELEN == 32 { c.expected_cause = 2; }
                cases.push(c);
            }
        }
        for scalar in [0x3f800000u64, 0x7f800123] {
            for (vl, start) in [(0, 0), (4, 0), (4, 1), (4, 4)] {
                let words = [
                    i(0x57, 5, 7, 4, 0xc00 | (2 << 3)),
                    (1 << 25) | (11 << 15) | (6 << 12) | (9 << 7) | 7,
                    i(7, 3, 2, 13, 0),
                    i(0x57, 5, 7, 12, (2 << 3) | 3),
                    i(0x73, 0, 5, start, 8),
                    r(0x57, 9, 5, 3, 0, 33),
                    i(0x57, 5, 7, 4, 0xc00 | (2 << 3)),
                    (1 << 25) | (10 << 15) | (6 << 12) | (9 << 7) | 0x27,
                    i(0x73, 7, 2, 0, 1),
                    s(0x23, 2, 10, 7, 64),
                    0x8067,
                ];
                let mut a = args;
                a[2] = vl;
                a[3] = 3 << 32;
                let mut c = micro(&format!("floating_insert_{scalar:x}_vl{vl}_start{start}"), &words, a);
                c.initial.put(3, 0, &scalar.to_le_bytes());
                cases.push(c);
            }
        }
        let mut a = args;
        a[2] = 17;
        cases.push(micro(
            "vset_avl_register_rules",
            &[
                i(0x57, 5, 7, 12, 16),
                s(0x23, 2, 10, 5, 0),
                i(0x57, 5, 7, 0, 16),
                s(0x23, 2, 10, 5, 4),
                i(0x57, 0, 7, 0, 16),
                i(0x73, 5, 2, 0, 0xc20),
                s(0x23, 2, 10, 5, 8),
                i(0x57, 5, 7, 3, 0xc10u32 as i32),
                s(0x23, 2, 10, 5, 12),
                0x8067,
            ],
            a,
        ));
        // SEW and integer/fractional LMUL, VL=0, mask/tail preservation, CSR visibility.
        for sew in 0..4 {
            for lmul in [0, 1, 2, 3, 5, 6, 7] {
                for avl in [0, 1, 17, 63] {
                    let vt = (sew << 3) | lmul;
                    let mut a = args;
                    a[2] = avl;
                    let width = [0, 5, 6, 7][sew as usize];
                    let words = [
                        i(0x57, 5, 7, 12, vt),
                        (1 << 25) | (11 << 15) | (width << 12) | (8 << 7) | 7,
                        r(0x57, 16, 0, 8, 8, 0) | (1 << 25),
                        (1 << 25) | (10 << 15) | (width << 12) | (16 << 7) | 0x27,
                        i(0x73, 6, 2, 0, 0xc20),
                        s(0x23, 2, 10, 6, 1536),
                        i(0x73, 6, 2, 0, 0xc21),
                        s(0x23, 2, 10, 6, 1540),
                        0x8067,
                    ];
                    let mut c = micro(
                        &format!("vector_sew{}_lmul{lmul}_avl{avl}", 8 << sew),
                        &words,
                        a,
                    );
                    if (8 << sew) > ELEN as i32 || (lmul >= 5 && (8 << sew) > (ELEN as i32 >> (8 - lmul))) {
                        c.expected_cause = 2;
                    }
                    cases.push(c);
                }
            }
        }
        for sew in 0..4 {
            let mut a = args;
            a[2] = 17;
            let width = [0, 5, 6, 7][sew as usize];
            let words = [
                i(0x57, 5, 7, 12, sew << 3),
                (1 << 25) | (11 << 15) | (width << 12) | (8 << 7) | 7,
                (0x18 << 26) | (1 << 25) | (8 << 20) | (3 << 12) | 0x57,
                (8 << 20) | (8 << 15) | (16 << 7) | 0x57,
                (1 << 25) | (10 << 15) | (width << 12) | (16 << 7) | 0x27,
                0x8067,
            ];
            let mut c = micro(&format!("masked_sew{}", 8 << sew), &words, a);
            if sew == 3 && ELEN == 32 { c.expected_cause = 2; }
            cases.push(c);
        }
        for sew in [2, 3] {
            let mut a = args;
            a[2] = 17;
            let width = if sew == 2 { 6 } else { 7 };
            let mut c = micro(
                &format!("vector_floating_{}", 8 << sew),
                &[
                    i(0x57, 5, 7, 12, sew << 3),
                    (1 << 25) | (11 << 15) | (width << 12) | (8 << 7) | 7,
                    (1 << 25) | (8 << 20) | (8 << 15) | (1 << 12) | (16 << 7) | 0x57,
                    (0x2c << 26) | (1 << 25) | (8 << 20) | (8 << 15) | (1 << 12) | (16 << 7) | 0x57,
                    (1 << 25) | (10 << 15) | (width << 12) | (16 << 7) | 0x27,
                    (0x12 << 26)
                        | (1 << 25)
                        | (16 << 20)
                        | (1 << 15)
                        | (1 << 12)
                        | (24 << 7)
                        | 0x57,
                    i(0x13, 7, 0, 10, 1024),
                    (1 << 25) | (7 << 15) | (width << 12) | (24 << 7) | 0x27,
                    i(0x73, 6, 2, 0, 1),
                    s(0x23, 2, 10, 6, 1536),
                    0x8067,
                ],
                a,
            );
            if sew == 3 {
                if ELEN == 32 { c.expected_cause = 2; }
                for n in 0..17 {
                    c.initial
                        .put(2, n * 8, &(n as f64 / 4.0 - 2.0).to_le_bytes());
                }
            }
            cases.push(c);
        }
        // Scalar IEEE arithmetic/FMA/conversion flags are stored through CSR reads.
        for fmt in 0..2 {
            let load_f = if fmt == 0 { 2 } else { 3 };
            let step = if fmt == 0 { 4 } else { 8 };
            let mut c = micro(
                &format!("floating_{}", 32 << fmt),
                &[
                    i(7, 1, load_f, 11, 0),
                    i(7, 2, load_f, 11, step),
                    r(0x53, 3, 0, 1, 2, fmt),
                    (2 << 27) | (fmt << 25) | (2 << 20) | (1 << 15) | (4 << 7) | 0x43,
                    s(0x27, load_f, 10, 3, 0),
                    s(0x27, load_f, 10, 4, 8),
                    r(0x53, 5, 0, 1, 0, 0x60 | fmt),
                    s(0x23, 2, 10, 5, 16),
                    i(0x73, 6, 2, 0, 1),
                    s(0x23, 2, 10, 6, 20),
                    0x8067,
                ],
                args,
            );
            if fmt == 0 {
                c.initial.put(
                    2,
                    0,
                    &[1.5f32.to_le_bytes(), (-2.25f32).to_le_bytes()].concat(),
                );
            } else {
                if ELEN == 32 { c.expected_cause = 2; }
                c.initial.put(
                    2,
                    0,
                    &[1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat(),
                );
            }
            cases.push(c);
        }
        // Raw IEEE encodings preserve signaling NaNs and signed zero in the inputs.
        for fmt in 0..2 {
            let fixtures: &[(&str, [u64; 3])] = if fmt == 0 {
                &[
                    ("zeros", [0, 0x80000000, 0x3f800000]),
                    ("division_by_zero", [0x3f800000, 0, 0xbf800000]),
                    ("negative_zero", [0x80000000, 0x40000000, 0]),
                    ("qnan", [0x7fc12345, 0x3f800000, 0xbf800000]),
                    ("snan", [0x7f812345, 0x3f800000, 0]),
                    ("infinities", [0x7f800000, 0, 0xff800000]),
                    ("subnormal", [1, 0x3f000000, 0x00800000]),
                    ("overflow", [0x7f7fffff, 0x40000000, 0xff7fffff]),
                    ("conversion_ties", [0x3fc00000, 0xc0200000, 0x3f800000]),
                    ("division_inexact", [0x3f800000, 0x40400000, 0x80000000]),
                    ("fused_cancellation", [0x3f800001, 0x3f800001, 0xbf800000]),
                ]
            } else {
                &[
                    ("zeros", [0, 0x8000000000000000, 0x3ff0000000000000]),
                    (
                        "division_by_zero",
                        [0x3ff0000000000000, 0, 0xbff0000000000000],
                    ),
                    ("negative_zero", [0x8000000000000000, 0x4000000000000000, 0]),
                    (
                        "qnan",
                        [0x7ff8123456789abc, 0x3ff0000000000000, 0xbff0000000000000],
                    ),
                    ("snan", [0x7ff0123456789abc, 0x3ff0000000000000, 0]),
                    ("infinities", [0x7ff0000000000000, 0, 0xfff0000000000000]),
                    ("subnormal", [1, 0x3fe0000000000000, 0x0010000000000000]),
                    (
                        "overflow",
                        [0x7fefffffffffffff, 0x4000000000000000, 0xffefffffffffffff],
                    ),
                    (
                        "conversion_ties",
                        [0x3ff8000000000000, 0xc004000000000000, 0x3ff0000000000000],
                    ),
                    (
                        "division_inexact",
                        [0x3ff0000000000000, 0x4008000000000000, 0x8000000000000000],
                    ),
                    (
                        "fused_cancellation",
                        [0x3ff0000000000001, 0x3ff0000000000001, 0xbff0000000000000],
                    ),
                    ("narrowing_tie", [0x3ff0000010000000, 0x3ff0000000000000, 0]),
                ]
            };
            let width = if fmt == 0 { 2 } else { 3 };
            let bytes = 1usize << width;
            for frm in 0..5 {
                for &(name, values) in fixtures {
                    if frm != 0
                        && !matches!(
                            name,
                            "conversion_ties" | "fused_cancellation" | "narrowing_tie"
                        )
                    {
                        continue;
                    }
                    let mut a = args;
                    a[2] = 0x01000001;
                    a[3] = 0xffffffff;
                    let mut words = vec![i(0x73, 0, 5, frm, 2)];
                    for f in 1..=3 {
                        words.push(i(7, f, width, 11, (f as usize - 1) as i32 * bytes as i32));
                    }
                    // Clear and sample fflags for every operation; result and flags share a 32-byte record.
                    let mut operations = vec![];
                    for op in [0, 4, 8, 12] {
                        operations.push((r(0x53, 4, 7, 1, 2, op | fmt), s(0x27, width, 10, 4, 0)));
                    }
                    for op in [0x43, 0x47, 0x4b, 0x4f] {
                        operations.push((
                            (3 << 27)
                                | (fmt << 25)
                                | (2 << 20)
                                | (1 << 15)
                                | (7 << 12)
                                | (4 << 7)
                                | op,
                            s(0x27, width, 10, 4, 0),
                        ));
                    }
                    for source in [1, 2] {
                        for unsigned in 0..2 {
                            operations.push((
                                r(0x53, 5, 7, source, unsigned, 0x60 | fmt),
                                s(0x23, 2, 10, 5, 0),
                            ));
                        }
                    }
                    for unsigned in 0..2 {
                        operations.push((
                            r(0x53, 4, 7, 12 + unsigned, unsigned, 0x68 | fmt),
                            s(0x27, width, 10, 4, 0),
                        ));
                    }
                    let target = 1 - fmt;
                    if fmt == 1 { operations.push((
                        r(0x53, 4, 7, 1, fmt, 0x20 | target),
                        s(0x27, if target == 0 { 2 } else { 3 }, 10, 4, 0),
                    )); }
                    for (n, (operation, store)) in operations.into_iter().enumerate() {
                        let offset = n as u32 * 32;
                        words.extend([
                            i(0x73, 0, 5, 0, 1),
                            operation,
                            store | ((offset >> 5) << 25),
                            i(0x73, 6, 2, 0, 1),
                            s(0x23, 2, 10, 6, offset + 16),
                        ]);
                    }
                    words.extend([i(0x73, 6, 2, 0, 2), s(0x23, 2, 10, 6, 1536), 0x8067]);
                    let mut c = micro(
                        &format!("scalar_ieee_{}_frm{frm}_{name}", 32 << fmt),
                        &words,
                        a,
                    );
                    if fmt == 1 && ELEN == 32 { c.expected_cause = 2; }
                    for (n, bits) in values.into_iter().enumerate() {
                        c.initial.put(2, n * bytes, &bits.to_le_bytes()[..bytes]);
                    }
                    cases.push(c);
                }
                // The vector case runs the same exceptional inputs lane by lane.
                let mut a = args;
                a[2] = fixtures.len() as u64;
                a[3] = 3 << 32;
                a[4] = 5 << 32;
                let vwidth: u32 = if fmt == 0 { 6 } else { 7 };
                let load =
                    |vd: u32, rs: u32| (1 << 25) | (rs << 15) | (vwidth << 12) | (vd << 7) | 7;
                let store = |vd: u32| (1 << 25) | (7 << 15) | (vwidth << 12) | (vd << 7) | 0x27;
                let mut words = vec![
                    i(0x73, 0, 5, frm, 2),
                    i(0x57, 5, 7, 12, (fmt as i32 + 2) << 3),
                    load(8, 11),
                    load(12, 13),
                ];
                for (n, operation) in [0, 2, 0x24, 0x20, 0x2c].into_iter().enumerate() {
                    let offset = n as u32 * 128;
                    words.extend([
                        i(0x73, 0, 5, 0, 1),
                        load(20, 14),
                        (operation << 26)
                            | (1 << 25)
                            | (12 << 20)
                            | (8 << 15)
                            | (1 << 12)
                            | (20 << 7)
                            | 0x57,
                        i(0x13, 7, 0, 10, offset as i32),
                        store(20),
                        i(0x73, 6, 2, 0, 1),
                        s(0x23, 2, 10, 6, 1024 + n as u32 * 4),
                    ]);
                }
                for unsigned in 0..2 {
                    words.extend([
                        i(0x73, 0, 5, 0, 1),
                        (0x12 << 26)
                            | (1 << 25)
                            | (8 << 20)
                            | ((1 - unsigned) << 15)
                            | (1 << 12)
                            | (24 << 7)
                            | 0x57,
                        i(0x13, 7, 0, 10, 640 + unsigned as i32 * 128),
                        store(24),
                        i(0x73, 6, 2, 0, 1),
                        s(0x23, 2, 10, 6, 1044 + unsigned * 4),
                    ]);
                }
                words.push(0x8067);
                let mut c = micro(&format!("vector_ieee_{}_frm{frm}", 32 << fmt), &words, a);
                if fmt == 1 && ELEN == 32 { c.expected_cause = 2; }
                for (n, (_, values)) in fixtures.iter().enumerate() {
                    for (bank, bits) in [(2, values[0]), (3, values[1]), (5, values[2])] {
                        c.initial.put(bank, n * bytes, &bits.to_le_bytes()[..bytes]);
                    }
                }
                cases.push(c);
            }
        }
        cases.push(micro(
            "bank_line_crossing_word",
            &[i(3, 5, 2, 11, 15), s(0x23, 2, 10, 5, 15), 0x8067],
            args,
        ));
        let mut double_load = micro(
            "bank_line_crossing_double",
            &[i(7, 5, 3, 11, 11), s(0x27, 3, 10, 5, 11), 0x8067],
            args,
        );
        if ELEN == 32 { double_load.expected_cause = 2; }
        cases.push(double_load);
        let mut a = args;
        a[2] = 17;
        let mut c = micro(
            "illegal_group_alignment",
            &[
                i(0x57, 5, 7, 12, 17),
                (1 << 25) | (11 << 15) | (6 << 12) | (9 << 7) | 7,
            ],
            a,
        );
        c.expected_cause = 2;
        cases.push(c);
        cases.push(micro("illegal_opcode", &[0xffffffff], args));
        cases.push(micro("load_fault", &[i(3, 5, 2, 11, 0)], {
            let mut a = args;
            a[1] = 6 << 32;
            a
        }));
        cases.push(micro("store_fault", &[s(0x23, 2, 10, 11, 0)], {
            let mut a = args;
            a[0] = (1 << 32) + 4094;
            a
        }));
        for (n, cause) in [
            (cases.len() - 3, 2),
            (cases.len() - 2, 5),
            (cases.len() - 1, 7),
        ] {
            cases[n].expected_cause = cause;
        }
        cases.extend(integer_depth_cases(args));
        let mut local = micro(
            "local_stack_and_constants",
            &[
                i(0x13, 2, 0, 2, -16),
                s(0x23, 2, 2, 11, 0),
                s(0x23, 2, 2, 12, 4),
                i(3, 5, 2, 2, 0),
                i(3, 6, 2, 2, 4),
                s(0x23, 2, 10, 5, 0),
                s(0x23, 2, 10, 6, 4),
                0x400003b7,
                i(3, 5, 2, 7, 0),
                s(0x23, 2, 10, 5, 8),
                i(0x13, 2, 0, 2, 16),
                0x8067,
            ],
            args,
        );
        local.constants[..4].copy_from_slice(&0x72cafe31u32.to_le_bytes());
        cases.push(local);
        let mut readonly = micro(
            "store_readonly_constants",
            &[0x400002b7, s(0x23, 2, 5, 11, 0), 0x8067],
            args,
        );
        readonly.expected_cause = 7;
        cases.push(readonly);
        for kernel in ["sin", "cos"] {
            let values: Vec<f32> = (0..65).map(|n| (n as f32 - 32.0) * 0.75).chain([
                120.0, -120.0, 10000.0, -10000.0, 1e10, -1e10, 1e30, -1e30,
                f32::MAX, -f32::MAX, f32::from_bits(1), -f32::from_bits(1),
                0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            ]).collect();
            let mut a = args;
            a[2] = values.len() as u64;
            a[3] = a[0] + 1024;
            let mut c = elf(&kernel_dir.join(format!("{kernel}.elf")), &format!("{kernel}_fp32_range"), a);
            for (n, value) in values.iter().enumerate() { c.initial.put(2, n * 4, &value.to_le_bytes()); }
            cases.push(c);
        }
        for kernel in ["gelu", "tanh"] {
            let mut a = args;
            a[2] = 37;
            a[3] = a[0] + 1024;
            cases.push(elf(&kernel_dir.join(format!("{kernel}.elf")), &format!("{kernel}_fp32"), a));
        }
        let b = initial();
        Self {
            cases,
            current: 0,
            actual: b.clone(),
            expected: b,
            engine: Engine::new(VLEN, ELEN, IBUF_BYTES),
            fault: None,
        }
    }
    pub fn select(&mut self, n: usize) {
        self.current = n;
        let c = &self.cases[n];
        self.actual = c.initial.clone();
        let descriptor = [
            u64::from(c.entry),
            c.program.len() as u64,
            0x40002000,
            u64::from(c.args[0]),
            u64::from(c.args[1]),
            u64::from(c.args[2]),
            u64::from(c.args[3]),
            u64::from(c.args[4]),
            u64::from(c.args[5]),
            u64::from(c.args[6]),
            u64::from(c.args[7]),
            0,
        ];
        for (index, value) in descriptor.iter().enumerate() {
            self.actual.put(1, 2048 + index * 8, &value.to_le_bytes());
        }
        self.expected = match &c.fixed_expected {
            Some(gold) => {
                let mut expected = gold.clone();
                let descriptor = BANK_BYTES + 2048..BANK_BYTES + 2144;
                expected.0[descriptor.clone()].copy_from_slice(&self.actual.0[descriptor]);
                expected
            }
            None => self.actual.clone(),
        };
        self.engine.load_constants(n % 2, &c.constants);
        self.engine.load_program(n % 2, &c.program);
        self.fault = if c.fixed_expected.is_some() { None } else { self
            .engine
            .run(
                n % 2,
                c.entry,
                c.program.len() as u32,
                c.args.map(u64::from),
                0x40002000,
                &mut self.expected,
            )
            .err() };
        assert_eq!(
            self.fault.map(|f| f.cause).unwrap_or(0),
            c.expected_cause,
            "unexpected reference outcome for {}: {:?}",
            c.name, self.fault
        );
        let read_float = |banks: &Banks, address: u64| {
            f32::from_le_bytes(banks.0[banks.range(address, 4).unwrap()].try_into().unwrap())
        };
        let operation = c.name.split('_').next().unwrap();
        if matches!(operation, "silu" | "swiglu" | "snake" | "sin" | "cos") {
            let length = c.args[if matches!(operation, "swiglu" | "snake") { 3 } else { 2 }] as usize;
            let channels = if operation == "snake" { c.args[2] as usize } else { 1 };
            for channel in 0..channels {
                for index in 0..length {
                    let offset = ((channel * length + index) * 4) as u64;
                    let x = read_float(&c.initial, c.args[1] + offset);
                    let expected = match operation {
                        "silu" => x / (1.0 + (-x).exp()),
                        "swiglu" => x / (1.0 + (-x).exp()) * read_float(&c.initial, c.args[2] + offset),
                        "snake" => {
                            let alpha = read_float(&c.initial, c.args[4] + channel as u64 * 4).exp();
                            let beta = read_float(&c.initial, c.args[5] + channel as u64 * 4).exp() + 1e-9;
                            let sine = (x * alpha).sin();
                            x + sine * sine / beta
                        }
                        "sin" => x.sin(),
                        "cos" => x.cos(),
                        _ => unreachable!(),
                    };
                    let actual = read_float(&self.expected, c.args[0] + offset);
                    assert!(if expected.is_nan() { actual.is_nan() } else {
                        actual == expected || (actual - expected).abs() <= 2e-6 * (1.0 + expected.abs())
                    }, "{} element {index}: {actual} != libm {expected}", c.name);
                }
            }
        }
        eprintln!("RVV case {n}: {}", c.name);
    }
}
