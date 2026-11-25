// Directed and random MXINT8MM cases. A case is a bank image loaded before
// its first command and a list of commands run back to back without reset;
// each command carries the C window the reference model expects afterwards.
use crate::reference::{execute, stride, Banks, Command};

pub const NUM_DIRECTED: u32 = 16;
pub const NUM_CASES: u32 = NUM_DIRECTED + 24;

pub struct Step {
    pub cmd: Command,
    // C window rows after the command, from bank row c_base.
    pub expect: Vec<u128>,
    pub features: u32,
}

pub struct Case {
    pub preload: Vec<((u32, u32, u32), u128)>,
    pub steps: Vec<Step>,
}

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    // Uniform in [lo, hi].
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next() % (hi - lo + 1)
    }

    fn codes(&mut self, rows: usize, k: usize) -> Vec<Vec<i8>> {
        let mode = self.next() % 4;
        (0..rows)
            .map(|_| {
                (0..k)
                    .map(|_| match mode {
                        0 => self.next() as i8,
                        1 => {
                            if self.next() % 10 < 7 {
                                0
                            } else {
                                self.next() as i8
                            }
                        }
                        2 => [-128, -127, 127, 0][(self.next() % 4) as usize],
                        _ => self.range(0, 8) as i8 - 4,
                    })
                    .collect()
            })
            .collect()
    }

    // Exponent bytes at most 177, so every term stays below 2^119.
    fn exps(&mut self, rows: usize, k: usize) -> Vec<Vec<u8>> {
        (0..rows)
            .map(|_| {
                (0..k / 32)
                    .map(|_| match self.next() % 10 {
                        0..=6 => self.range(119, 135) as u8,
                        7 => self.range(0, 40) as u8,
                        8 => self.range(0, 177) as u8,
                        _ => self.range(128, 177) as u8,
                    })
                    .collect()
            })
            .collect()
    }

    // Finite old C: normals below 2^30, signed zeros and subnormals.
    fn old(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|_| match self.next() % 8 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(self.next() & 0x807f_ffff),
                _ => f32::from_bits((self.next() & 0x807f_ffff) | self.range(107, 156) << 23),
            })
            .collect()
    }
}

struct Builder {
    rng: Rng,
    entries: usize,
    banks: Banks,
    steps: Vec<Step>,
}

impl Builder {
    // Packed MXINT8 rows from bank row `base`: the codes, then the exponent bytes.
    fn place(&mut self, at: (u32, u32), base: usize, codes: &[Vec<i8>], exps: &[Vec<u8>]) {
        let k = codes[0].len();
        let s = stride(k);
        for (r, (codes, exps)) in codes.iter().zip(exps).enumerate() {
            assert!(codes.len() == k && exps.len() == k / 32, "mxint8matmul: ragged operand");
            let mut bytes: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
            bytes.extend_from_slice(exps);
            bytes.resize(s * 16, 0);
            for (o, chunk) in bytes.chunks_exact(16).enumerate() {
                let row = (base + r * s + o) as u32;
                self.banks
                    .insert((at.0, at.1, row), u128::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
    }

    fn place_c(&mut self, at: (u32, u32), base: usize, values: &[f32]) {
        for (o, words) in values.chunks_exact(4).enumerate() {
            let mut data = 0u128;
            for (w, value) in words.iter().enumerate() {
                data |= u128::from(value.to_bits()) << (32 * w);
            }
            self.banks.insert((at.0, at.1, (base + o) as u32), data);
        }
    }

    // Place random A, B (and old C when accumulating) for `cmd`.
    fn random_operands(&mut self, cmd: &Command) {
        let (a_codes, a_exps) = (self.rng.codes(cmd.m, cmd.k), self.rng.exps(cmd.m, cmd.k));
        let (b_codes, b_exps) = (self.rng.codes(cmd.n, cmd.k), self.rng.exps(cmd.n, cmd.k));
        self.place(cmd.a, cmd.a_base, &a_codes, &a_exps);
        self.place(cmd.b, cmd.b_base, &b_codes, &b_exps);
        if cmd.accumulate {
            let old = self.rng.old(cmd.m * cmd.n);
            self.place_c((cmd.read_bank, cmd.c.1), cmd.c_base, &old);
        }
    }

    fn run(&mut self, cmd: Command) {
        let features = execute(&mut self.banks, self.entries, &cmd);
        let expect = (0..cmd.m * cmd.n / 4)
            .map(|o| self.banks[&(cmd.c.0, cmd.c.1, (cmd.c_base + o) as u32)])
            .collect();
        self.steps.push(Step { cmd, expect, features });
    }
}

fn command(m: usize, n: usize, k: usize, bases: [usize; 3], accumulate: bool) -> Command {
    Command {
        a: (2, 5),
        b: (7, 1),
        c: (11, 9),
        read_bank: 13,
        m,
        n,
        k,
        a_base: bases[0],
        b_base: bases[1],
        c_base: bases[2],
        accumulate,
    }
}

pub fn gen_case(index: u32, entries: usize) -> Case {
    let mut b = Builder {
        rng: Rng(0x2545_f491 ^ index.wrapping_mul(0x9e37_79b9)),
        entries,
        banks: Banks::new(),
        steps: Vec::new(),
    };
    // The case's commands; their operands are placed before any command runs.
    let cmds = match index {
        // One 1x4 output tile, K = 32, all bases 0.
        0 => vec![command(1, 4, 32, [0, 0, 0], false)],
        // Partial row and column tiles, nonzero bases.
        1 => vec![command(5, 12, 64, [1, 2, 3], false)],
        // Accumulate onto an old C read from a renamed bank.
        2 => vec![command(6, 16, 96, [4, 0, 7], true)],
        // Accumulate onto the old C of the destination bank itself.
        3 => vec![Command {
            read_bank: 11,
            ..command(4, 8, 32, [0, 5, 0], true)
        }],
        // K = 544 (two exponent rows), bases 3/1/2, then accumulated onto itself.
        4 => {
            let cmd = command(3, 8, 544, [3, 1, 2], false);
            b.random_operands(&cmd);
            vec![
                cmd.clone(),
                Command {
                    read_bank: 11,
                    accumulate: true,
                    ..cmd
                },
            ]
        }
        // K = 512 (one full exponent row), then accumulated into a renamed bank.
        5 => {
            let cmd = command(4, 4, 512, [0, 0, 0], false);
            b.random_operands(&cmd);
            vec![
                cmd.clone(),
                Command {
                    c: (13, 9),
                    read_bank: 11,
                    accumulate: true,
                    ..cmd
                },
            ]
        }
        // K = 1024, then accumulated onto itself.
        6 => {
            let cmd = command(4, 8, 1024, [0, 0, 0], false);
            b.random_operands(&cmd);
            vec![
                cmd.clone(),
                Command {
                    read_bank: 11,
                    accumulate: true,
                    ..cmd
                },
            ]
        }
        // Subnormal terms: one product per block at 2^-150 and 2^-151, so the
        // conversion rounds, ties to even both ways and flushes negatives to -0.0.
        7 => {
            let cmd = command(4, 4, 64, [0, 0, 0], false);
            let lane = |values: [i8; 4], row: usize| {
                let mut codes = vec![0i8; 64];
                codes[0] = values[row];
                codes[32] = values[row];
                codes
            };
            let a: Vec<Vec<i8>> = (0..4).map(|r| lane([1, 3, -1, 5], r)).collect();
            let bb: Vec<Vec<i8>> = (0..4).map(|r| lane([1, 1, 3, -7], r)).collect();
            b.place(cmd.a, 0, &a, &vec![vec![52, 52]; 4]);
            b.place(cmd.b, 0, &bb, &vec![vec![52, 51]; 4]);
            vec![cmd]
        }
        // Exponent bytes 0 flush every term to a signed zero; the old C holds
        // -0.0, +0.0 and subnormals, so results keep -0.0 and subnormals.
        8 => {
            let cmd = command(4, 8, 64, [0, 0, 0], true);
            let mut a = b.rng.codes(4, 64);
            let mut bb = b.rng.codes(8, 64);
            a[0] = vec![1; 64];
            bb[0] = vec![-1; 64];
            b.place(cmd.a, 0, &a, &vec![vec![0, 0]; 4]);
            b.place(cmd.b, 0, &bb, &vec![vec![0, 0]; 8]);
            let old: Vec<f32> = (0..32)
                .map(|i| [-0.0, 0.0, f32::from_bits(1), f32::from_bits(0x8000_0003), 1.5][i % 5])
                .collect();
            b.place_c((cmd.read_bank, cmd.c.1), 0, &old);
            vec![cmd]
        }
        // Extreme products 2^19 cancelled exactly by -2^18 * 2 in the next block.
        9 => {
            let cmd = command(4, 8, 64, [0, 0, 0], false);
            let mut a = b.rng.codes(4, 64);
            let mut bb = b.rng.codes(8, 64);
            let mut a_exps = b.rng.exps(4, 64);
            let mut b_exps = b.rng.exps(8, 64);
            for r in 0..2 {
                a[r] = vec![-128; 64];
                a_exps[r] = vec![127, 127];
            }
            for r in 0..4 {
                bb[r] = [vec![-128; 32], vec![64; 32]].concat();
                b_exps[r] = vec![127, 128];
            }
            b.place(cmd.a, 0, &a, &a_exps);
            b.place(cmd.b, 0, &bb, &b_exps);
            vec![cmd]
        }
        // Large terms: exponent bytes 255 and 177 against small partners.
        10 => {
            let cmd = command(4, 8, 128, [0, 0, 0], false);
            let a = b.rng.codes(4, 128);
            let bb = b.rng.codes(8, 128);
            let a_exps: Vec<Vec<u8>> = (0..4).map(|r| vec![[255, 177, 200, 140][r]; 4]).collect();
            let b_exps: Vec<Vec<u8>> = (0..8)
                .map(|_| (0..4).map(|_| b.rng.range(60, 99) as u8).collect())
                .collect();
            b.place(cmd.a, 0, &a, &a_exps);
            b.place(cmd.b, 0, &bb, &b_exps);
            vec![cmd]
        }
        // Every window ends at the last bank row.
        11 => vec![command(8, 8, 32, [entries - 24, entries - 24, entries - 16], true)],
        // Many row tiles: M = 256, N = 4.
        12 => vec![command(256, 4, 32, [0, 3, 0], false)],
        // Many column tiles: N = 128.
        13 => vec![command(2, 128, 64, [9, 0, 5], false)],
        // K = 96 split over three commands: overwrite, accumulate from a renamed
        // bank, accumulate in place. The chain must equal the one-command product.
        14 => {
            let (m, n) = (5, 12);
            let a = b.rng.codes(m, 96);
            let bb = b.rng.codes(n, 96);
            let a_exps = b.rng.exps(m, 96);
            let b_exps = b.rng.exps(n, 96);
            let mut cmds = Vec::new();
            for s in 0..3 {
                let cmd = Command {
                    c: (if s == 0 { 11 } else { 12 }, 9),
                    read_bank: if s == 2 { 12 } else { 11 },
                    ..command(m, n, 32, [100 * s, 100 * s, 0], s > 0)
                };
                let codes: Vec<Vec<i8>> = a.iter().map(|r| r[32 * s..32 * s + 32].to_vec()).collect();
                let exps: Vec<Vec<u8>> = a_exps.iter().map(|r| vec![r[s]]).collect();
                b.place(cmd.a, cmd.a_base, &codes, &exps);
                let codes: Vec<Vec<i8>> = bb.iter().map(|r| r[32 * s..32 * s + 32].to_vec()).collect();
                let exps: Vec<Vec<u8>> = b_exps.iter().map(|r| vec![r[s]]).collect();
                b.place(cmd.b, cmd.b_base, &codes, &exps);
                cmds.push(cmd);
            }
            let whole = command(m, n, 96, [0, 0, 0], false);
            let mut one = Builder {
                rng: Rng(1),
                entries,
                banks: Banks::new(),
                steps: Vec::new(),
            };
            one.place(whole.a, 0, &a, &a_exps);
            one.place(whole.b, 0, &bb, &b_exps);
            one.run(whole);
            let chain = cmds.clone();
            let preload: Vec<_> = b.banks.clone().into_iter().collect();
            for cmd in chain {
                b.run(cmd);
            }
            assert_eq!(
                b.steps[2].expect, one.steps[0].expect,
                "mxint8matmul: K split differs from one command"
            );
            return Case {
                preload,
                steps: b.steps,
            };
        }
        // Four independent commands back to back on different banks.
        15 => vec![
            command(3, 8, 64, [0, 0, 0], false),
            Command {
                a: (0, 3),
                b: (1, 4),
                c: (3, 2),
                read_bank: 4,
                ..command(9, 20, 32, [6, 2, 0], true)
            },
            Command {
                a: (20, 0),
                b: (21, 23),
                c: (22, 0),
                read_bank: 23,
                ..command(4, 20, 160, [1, 1, 1], false)
            },
            Command {
                a: (5, 0),
                b: (6, 1),
                c: (8, 2),
                read_bank: 10,
                ..command(4, 32, 64, [0, 0, 0], true)
            },
        ],
        // Random shapes, bases, banks, groups, operands and old C.
        _ => {
            let k = 32 * b.rng.range(1, 24) as usize;
            let s = stride(k);
            let m = b.rng.range(1, 24.min(entries / s) as u32) as usize;
            let n = 4 * b.rng.range(1, 12.min(entries / s / 4) as u32) as usize;
            let mut ids: Vec<u32> = (0..24).collect();
            for i in 0..4 {
                let j = b.rng.range(i as u32, 23) as usize;
                ids.swap(i, j);
            }
            vec![Command {
                a: (ids[0], b.rng.range(0, 23)),
                b: (ids[1], b.rng.range(0, 23)),
                c: (ids[2], b.rng.range(0, 23)),
                read_bank: if b.rng.next() % 3 == 0 { ids[2] } else { ids[3] },
                m,
                n,
                k,
                a_base: b.rng.range(0, (entries - m * s) as u32) as usize,
                b_base: b.rng.range(0, (entries - n * s) as u32) as usize,
                c_base: b.rng.range(0, (entries - m * n / 4) as u32) as usize,
                accumulate: b.rng.next() % 2 == 1,
            }]
        }
    };
    // Cases 4 to 10 place their operands above; the others get random A, B and old C.
    if !(4..=10).contains(&index) {
        for cmd in &cmds {
            b.random_operands(cmd);
        }
    }
    let preload = b.banks.clone().into_iter().collect();
    for cmd in cmds {
        b.run(cmd);
    }
    Case {
        preload,
        steps: b.steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_cover_every_feature() {
        let mut features = 0;
        for index in 0..NUM_CASES {
            for step in gen_case(index, 1024).steps {
                features |= step.features;
            }
        }
        // Every semantic event of reference.rs occurs in some case.
        assert_eq!(features, (1 << 15) - 1);
    }
}
