use crate::model::{self, GROUPS, LANE};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecMat16Cmd {
    pub iter: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub wr_read_bank: u32,
    pub rob_id: u32,
    // Group bases; the old C uses the wr layout, so it shares wr_group.
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
}

#[derive(Clone, Debug)]
pub struct VecMat16Case {
    pub cmd: VecMat16Cmd,
    pub at: Vec<u128>,
    pub b: Vec<u128>,
    // Indexed group * LANE + row.
    pub c_old: Vec<u128>,
    pub c_new: Vec<u128>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

#[derive(Clone, Copy)]
enum Fill {
    Random,
    // Every int8 is -128 or 127: the largest products of both signs.
    Extremes,
}

fn i8_word(rng: &mut Lcg, fill: Fill) -> u128 {
    let mut word = 0u128;
    for lane in 0..LANE {
        let byte = match fill {
            Fill::Random => rng.next() as u8,
            Fill::Extremes => {
                if rng.next() & 1 == 0 {
                    0x80
                } else {
                    0x7f
                }
            }
        };
        word |= u128::from(byte) << (8 * lane);
    }
    word
}

pub fn gen_case(index: u32) -> VecMat16Case {
    let entries = model::bank_entries();
    // (rob_id, iter, [op1, op2, wr, wr_read], [op1, op2, wr] group bases, fill,
    //  old C near the int32 limits)
    let (rob_id, iter, banks, groups, fill, wrap) = match index {
        0 => (1, 1, [0, 1, 2, 3], [0, 0, 0], Fill::Random, false),
        1 => (2, 16, [4, 5, 6, 7], [0, 0, 0], Fill::Extremes, false),
        2 => (3, 5, [1, 0, 3, 2], [0, 0, 0], Fill::Extremes, true),
        3 => (4, entries, [8, 9, 10, 11], [0, 0, 0], Fill::Random, false),
        4 => (5, 7, [1023, 512, 700, 701], [0, 0, 0], Fill::Random, true),
        5 => (6, 9, [12, 13, 14, 15], [3, 7, 12], Fill::Random, false),
        6 => (7, 16, [20, 20, 21, 22], [1, 9, 5], Fill::Extremes, true),
        _ => panic!("vecmat16: unsupported case {index}"),
    };
    model::validate(entries, iter);
    let mut rng = Lcg(0x5ec0_0000 + u64::from(index));
    let at: Vec<u128> = (0..iter).map(|_| i8_word(&mut rng, fill)).collect();
    let b: Vec<u128> = (0..iter).map(|_| i8_word(&mut rng, fill)).collect();
    let mut c = [[0i32; LANE]; LANE];
    for row in c.iter_mut() {
        for value in row.iter_mut() {
            let r = rng.next();
            *value = if wrap {
                // Near i32::MAX or i32::MIN so the accumulation wraps.
                if r & 1 == 0 {
                    i32::MAX - (r >> 20) as i32
                } else {
                    i32::MIN + (r >> 20) as i32
                }
            } else {
                r as i32
            };
        }
    }
    let mut c_old = Vec::with_capacity(GROUPS * LANE);
    for g in 0..GROUPS {
        for row in 0..LANE {
            c_old.push(model::c_word(&c, g, row));
        }
    }
    model::vecmat16(&at, &b, &mut c);
    let mut c_new = Vec::with_capacity(GROUPS * LANE);
    for g in 0..GROUPS {
        for row in 0..LANE {
            c_new.push(model::c_word(&c, g, row));
        }
    }
    VecMat16Case {
        cmd: VecMat16Cmd {
            iter,
            op1_bank: banks[0],
            op2_bank: banks[1],
            wr_bank: banks[2],
            wr_read_bank: banks[3],
            rob_id,
            op1_group: groups[0],
            op2_group: groups[1],
            wr_group: groups[2],
        },
        at,
        b,
        c_old,
        c_new,
    }
}
