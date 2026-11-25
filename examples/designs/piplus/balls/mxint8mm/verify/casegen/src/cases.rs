//! MXINT8MM cases: each a few commands back to back on one reset, against a
//! bank image per (bank, group) that mirrors the testbench memory, so that a
//! command can accumulate onto the C an earlier command wrote.
use crate::reference::{execute, packed_rows, Command, ROW_BYTES};
use std::collections::HashMap;

pub const DIRECTED_CASES: u32 = 17;
pub const RANDOM_CASES: u32 = 14;
pub const NUM_CASES: u32 = DIRECTED_CASES + RANDOM_CASES;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Mxint8mmCmd {
    pub rob_id: u32,
    pub iter: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub a_bank: u32,
    pub b_bank: u32,
    pub c_bank: u32,
    pub read_bank: u32,
    pub a_group: u32,
    pub b_group: u32,
    pub c_group: u32,
    pub update: u32,
    pub features: u32,
    pub stall: u32,
}

#[derive(Clone, Debug)]
pub struct Preload {
    pub bank: u32,
    pub group: u32,
    pub addr: u32,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub cmd: Mxint8mmCmd,
    pub preload: Vec<Preload>,
    pub writes: Vec<(u32, u128)>,
}

#[derive(Clone, Debug)]
pub struct Case {
    pub steps: Vec<Step>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Values {
    // Any code, block scales near one.
    Random,
    // Codes -128, 127, -127, 0, 1, -1 and exponent bytes 0, 1, 255 and others.
    Extremes,
    // Block scales 2^-170 .. 2^-130: normal, subnormal and flushed terms.
    Subnormal,
    // Codes 100 .. 127 and block scales 2^87 .. 2^104: terms up to 2^123.
    Huge,
    // Even blocks whose products cancel to p = 0.
    Cancel,
    // Every code -128 and scale 2^108: each term 2^127.
    Max,
}

struct Spec {
    cmd: Command,
    values: Values,
    // Banks and groups of A, B and C, and the bank the old C is read from.
    banks: [u32; 3],
    groups: [u32; 3],
    read_bank: u32,
    // A new old C window when accumulating; otherwise the C an earlier
    // command of the case wrote.
    fill_old: bool,
    // Old C and bias values -0.0, +0.0 and subnormals among normal values.
    special: bool,
    stall: bool,
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn below(&mut self, limit: u32) -> u32 {
        self.next() % limit
    }

    // A normal FP32 value of magnitude 2^-20 .. 2^21.
    fn float(&mut self, special: bool) -> f32 {
        if special && self.below(2) == 0 {
            let bits = [0x8000_0000, 0x0000_0000, 0x0000_0001, 0x807f_ffff, 0x0040_0000];
            return f32::from_bits(bits[self.below(5) as usize]);
        }
        let exponent = 107 + self.below(41);
        f32::from_bits((self.below(2) << 31) | (exponent << 23) | self.below(1 << 23))
    }
}

// One packed MXINT8 row: k codes, k / 32 exponent bytes, padded to stride rows.
fn packed_row(rng: &mut Rng, values: Values, lhs: bool, k: usize) -> Vec<u8> {
    let mut row = vec![0u8; packed_rows(k) * ROW_BYTES];
    for block in 0..k / 32 {
        let codes = &mut row[block * 32..block * 32 + 32];
        for i in 0..32 {
            codes[i] = match values {
                Values::Extremes => [0x80, 0x7f, 0x81, 0x00, 0x01, 0xff][rng.below(6) as usize],
                Values::Huge => (100 + rng.below(28)) as u8,
                Values::Max => 0x80,
                _ => rng.next() as u8,
            };
        }
        if values == Values::Cancel && block % 2 == 0 {
            // A pairs x, x and B pairs y, -y: every pair of products cancels.
            for i in (0..32).step_by(2) {
                if lhs {
                    codes[i + 1] = codes[i];
                } else {
                    let y = (rng.below(127) + 1) as i8 * if rng.below(2) == 0 { 1 } else { -1 };
                    codes[i] = y as u8;
                    codes[i + 1] = (-y) as u8;
                }
            }
        }
        row[k + block] = match (values, lhs) {
            (Values::Random | Values::Cancel, _) => (119 + rng.below(17)) as u8,
            (Values::Extremes, true) => [255, 0, 127, 200][rng.below(4) as usize],
            (Values::Extremes, false) => [0, 1, 60, 100][rng.below(4) as usize],
            (Values::Subnormal, true) => rng.below(21) as u8,
            (Values::Subnormal, false) => (84 + rng.below(21)) as u8,
            (Values::Huge, true) => (200 + rng.below(9)) as u8,
            (Values::Huge, false) => (141 + rng.below(10)) as u8,
            (Values::Max, _) => 181,
        };
    }
    row
}

fn image<'a>(images: &'a mut HashMap<(u32, u32), Vec<u8>>, entries: usize, bank: u32, group: u32) -> &'a mut Vec<u8> {
    images
        .entry((bank, group))
        .or_insert_with(|| vec![0; entries * ROW_BYTES])
}

fn word(bytes: &[u8]) -> u128 {
    u128::from_le_bytes(bytes.try_into().unwrap())
}

// Bank rows from `addr` hold `bytes`: into the image and the preload list.
fn put(
    images: &mut HashMap<(u32, u32), Vec<u8>>,
    preload: &mut Vec<Preload>,
    entries: usize,
    (bank, group): (u32, u32),
    addr: usize,
    bytes: &[u8],
) {
    let target = image(images, entries, bank, group);
    target[addr * ROW_BYTES..addr * ROW_BYTES + bytes.len()].copy_from_slice(bytes);
    for (i, chunk) in bytes.chunks_exact(ROW_BYTES).enumerate() {
        preload.push(Preload {
            bank,
            group,
            addr: (addr + i) as u32,
            data: word(chunk),
        });
    }
}

fn floats(rng: &mut Rng, count: usize, special: bool) -> Vec<u8> {
    (0..count).flat_map(|_| rng.float(special).to_le_bytes()).collect()
}

fn step(rng: &mut Rng, images: &mut HashMap<(u32, u32), Vec<u8>>, entries: usize, rob_id: u32, s: Spec) -> Step {
    let c = s.cmd;
    assert!(c.fits(entries), "mxint8mm case: command exceeds the bank: {c:?}");
    let stride = packed_rows(c.k);
    let [a, b, cb] = s.banks;
    let [ag, bg, cg] = s.groups;
    let read_old = c.accumulate || c.bias;
    assert!(a != b && a != cb && b != cb && s.read_bank != a && s.read_bank != b);
    let mut preload = Vec::new();
    for row in 0..c.m {
        let bytes = packed_row(rng, s.values, true, c.k);
        put(images, &mut preload, entries, (a, ag), c.a_base + row * stride, &bytes);
    }
    for col in 0..c.n {
        let bytes = packed_row(rng, s.values, false, c.k);
        put(images, &mut preload, entries, (b, bg), col * stride, &bytes);
    }
    if c.accumulate && s.fill_old {
        for row in 0..c.m {
            let bytes = floats(rng, c.n, s.special);
            put(
                images,
                &mut preload,
                entries,
                (s.read_bank, cg),
                c.c_base + row * c.c_pitch,
                &bytes,
            );
        }
    }
    if c.bias {
        let bytes = floats(rng, c.n, s.special);
        put(
            images,
            &mut preload,
            entries,
            (s.read_bank, cg),
            c.c_base - c.n / 4,
            &bytes,
        );
    }
    let old = image(images, entries, s.read_bank, cg).clone();
    let lhs = image(images, entries, a, ag).clone();
    let rhs = image(images, entries, b, bg).clone();
    let (result, features) = execute(&c, &lhs, &rhs, &old);
    let target = image(images, entries, cb, cg);
    let mut writes = Vec::new();
    for (row, values) in result.chunks_exact(c.n).enumerate() {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        for (group, chunk) in bytes.chunks_exact(ROW_BYTES).enumerate() {
            let addr = c.c_base + row * c.c_pitch + group;
            target[addr * ROW_BYTES..(addr + 1) * ROW_BYTES].copy_from_slice(chunk);
            writes.push((addr as u32, word(chunk)));
        }
    }
    let rs2 = c.rs2();
    Step {
        cmd: Mxint8mmCmd {
            rob_id,
            iter: c.k as u32,
            rs2_lo: rs2 as u32,
            rs2_hi: (rs2 >> 32) as u32,
            a_bank: a,
            b_bank: b,
            c_bank: cb,
            read_bank: s.read_bank,
            a_group: ag,
            b_group: bg,
            c_group: cg,
            update: u32::from(read_old && s.read_bank == cb),
            features,
            stall: u32::from(s.stall),
        },
        preload,
        writes,
    }
}

fn command(
    m: usize,
    n: usize,
    k: usize,
    a_base: usize,
    c_pitch: usize,
    c_base: usize,
    accumulate: bool,
    bias: bool,
) -> Command {
    Command {
        m,
        n,
        k,
        a_base,
        c_pitch,
        c_base,
        accumulate,
        bias,
    }
}

// A, B, C in banks 2, 5, 9 (groups 1, 3, 4); the old C from bank 11 (renamed
// destination) or from bank 9 (update).
fn spec(cmd: Command, values: Values, update: bool) -> Spec {
    Spec {
        cmd,
        values,
        banks: [2, 5, 9],
        groups: [1, 3, 4],
        read_bank: if update { 9 } else { 11 },
        fill_old: true,
        special: false,
        stall: false,
    }
}

fn random_spec(rng: &mut Rng, entries: usize) -> Spec {
    let cmd = loop {
        let n = 4 * (1 + rng.below(16) as usize);
        let bias = rng.below(2) == 1;
        let cmd = Command {
            m: 1 + rng.below(40) as usize,
            n,
            k: 32 * (1 + rng.below(20) as usize),
            a_base: rng.below(8) as usize,
            c_pitch: n / 4 + rng.below(3) as usize,
            c_base: if bias { n / 4 } else { 0 } + rng.below(8) as usize,
            accumulate: rng.below(2) == 1,
            bias,
        };
        if cmd.fits(entries) {
            break cmd;
        }
    };
    let values = [
        Values::Random,
        Values::Random,
        Values::Subnormal,
        Values::Extremes,
        Values::Cancel,
    ][rng.below(5) as usize];
    // Four distinct banks: A, B, C and the old C (renamed) or nothing read.
    let mut banks = Vec::new();
    while banks.len() < 4 {
        let bank = rng.below(24);
        if !banks.contains(&bank) {
            banks.push(bank);
        }
    }
    let update = rng.below(2) == 1;
    Spec {
        cmd,
        values,
        banks: [banks[0], banks[1], banks[2]],
        groups: [rng.below(24), rng.below(24), rng.below(24)],
        read_bank: if update { banks[2] } else { banks[3] },
        fill_old: true,
        special: rng.below(3) == 0,
        stall: false,
    }
}

pub fn gen_case(index: u32, bank_entries: u32, rob_entries: u32) -> Case {
    let entries = bank_entries as usize;
    let mut rng = Rng(0x2545_f491 ^ index.wrapping_mul(0x9e37_79b9));
    let mut images = HashMap::new();
    let mut specs = Vec::new();
    match index {
        // One full tile, one block.
        0 => specs.push(spec(command(8, 8, 32, 0, 2, 0, false, false), Values::Random, false)),
        // Partial tile, one column group, A and C off row 0.
        1 => specs.push(spec(command(3, 4, 64, 5, 1, 7, false, false), Values::Random, false)),
        // Accumulate onto a renamed destination; C rows with gaps.
        2 => specs.push(spec(command(13, 20, 96, 2, 7, 3, true, false), Values::Random, false)),
        // Bias only; partial column tile.
        3 => specs.push(spec(command(9, 12, 64, 0, 3, 10, false, true), Values::Random, false)),
        // Accumulate and bias in place.
        4 => specs.push(spec(command(18, 36, 64, 1, 9, 9, true, true), Values::Random, true)),
        // 16 blocks share one exponent row; two full tile rows.
        5 => specs.push(spec(command(16, 8, 512, 0, 2, 0, false, false), Values::Random, false)),
        // 17 blocks: the second exponent row.
        6 => specs.push(spec(command(5, 8, 544, 3, 2, 1, true, false), Values::Random, false)),
        // Extreme codes and exponent bytes, with bias.
        7 => specs.push(spec(command(8, 16, 128, 0, 4, 4, false, true), Values::Extremes, false)),
        // Subnormal and flushed terms onto special old C values.
        8 => {
            let mut s = spec(command(6, 8, 64, 0, 2, 2, true, true), Values::Subnormal, false);
            s.special = true;
            specs.push(s);
        }
        // Terms of 2^105 .. 2^123.
        9 => specs.push(spec(command(4, 8, 64, 0, 2, 0, false, false), Values::Huge, false)),
        // Every term 2^127, the largest power of two below the FP32 limit.
        10 => specs.push(spec(command(8, 8, 32, 0, 2, 0, false, false), Values::Max, false)),
        // Blocks that cancel to p = 0 with nonzero products.
        11 => specs.push(spec(command(8, 8, 64, 0, 2, 0, true, false), Values::Cancel, false)),
        // A chain: 3 blocks into bank 9; 1 block from bank 9 into bank 12
        // (renamed); 2 blocks with bias in place in bank 12. Odd block counts
        // leave the array's slot pointer at 1 between commands.
        12 => {
            specs.push(spec(command(8, 12, 96, 0, 3, 4, false, false), Values::Random, false));
            let mut s = spec(command(8, 12, 32, 0, 3, 4, true, false), Values::Random, false);
            s.banks[2] = 12;
            s.read_bank = 9;
            s.fill_old = false;
            specs.push(s);
            let mut s = spec(command(8, 12, 64, 0, 3, 4, true, true), Values::Random, true);
            s.banks[2] = 12;
            s.read_bank = 12;
            s.fill_old = false;
            specs.push(s);
        }
        // Response held back by cmdResp.ready.
        13 => {
            let mut s = spec(command(2, 4, 32, 0, 1, 0, false, false), Values::Random, false);
            s.stall = true;
            specs.push(s);
        }
        // A and C windows end at the last bank row.
        14 => {
            let m = (entries - 1) / 3;
            let c_base = entries - (m - 1) * 3 - 1;
            specs.push(spec(
                command(m, 4, 32, entries - 3 * m, 3, c_base, true, false),
                Values::Random,
                true,
            ));
        }
        // One C row of the widest B the bank holds, with bias in place.
        15 => {
            let n = entries / 3 / 4 * 4;
            specs.push(spec(
                command(1, n, 32, 0, n / 4, entries - n / 4, false, true),
                Values::Random,
                true,
            ));
        }
        // One block from each start value: the tile's first block is also
        // its last. Old C, bias and both, renamed and in place.
        16 => {
            for (accumulate, bias) in [(true, false), (false, true), (true, true)] {
                for update in [false, true] {
                    specs.push(spec(
                        command(10, 12, 32, 1, 4, 5, accumulate, bias),
                        Values::Random,
                        update,
                    ));
                }
            }
        }
        _ => {
            for _ in 0..1 + rng.below(3) {
                specs.push(random_spec(&mut rng, entries));
            }
        }
    }
    let steps = specs
        .into_iter()
        .map(|s| {
            let rob_id = rng.below(rob_entries);
            step(&mut rng, &mut images, entries, rob_id, s)
        })
        .collect();
    Case { steps }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::*;

    #[test]
    fn cases_build_and_cover() {
        let mut features = 0;
        // {accumulate, bias, update} x {one block, 2 .. 15 blocks}
        let mut starts = [[false; 2]; 8];
        for index in 0..NUM_CASES {
            for step in gen_case(index, 1024, 8).steps {
                features |= step.cmd.features;
                let (accumulate, bias) = ((step.cmd.rs2_hi >> 30) & 1, step.cmd.rs2_hi >> 31);
                let start = (accumulate << 2 | bias << 1 | step.cmd.update) as usize;
                let blocks = step.cmd.iter / 32;
                if blocks < 16 {
                    starts[start][usize::from(blocks > 1)] = true;
                }
            }
        }
        for start in [0, 2, 3, 4, 5, 6, 7] {
            assert_eq!(
                starts[start],
                [true, true],
                "start {start:03b} not covered by one and few blocks"
            );
        }
        for feature in [
            F_SUBNORMAL_TERM,
            F_FLUSHED_TERM,
            F_HUGE_TERM,
            F_CANCELLED_BLOCK,
            F_CODE_MIN,
            F_EXPONENT_MAX,
            F_EXPONENT_MIN,
            F_INEXACT_ADD,
            F_SPECIAL_START,
        ] {
            assert_ne!(features & feature, 0, "feature {feature:#x} not covered");
        }
    }
}
