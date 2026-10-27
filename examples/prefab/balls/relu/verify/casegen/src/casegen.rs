use crate::model;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReluCmd {
    pub iter: u32,
    pub stride: u32,
    pub bank: u32,
    pub group_base: u32,
    pub group: u32,
    pub cols: u32,
    pub rob_id: u32,
    pub num_words: u32,
}

#[derive(Clone, Debug)]
pub struct ReluCase {
    pub cmd: ReluCmd,
    pub rows: Vec<u32>,
    pub src: Vec<u128>,
    pub dst: Vec<u128>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

// Lane values cycle through the edges (zero, -1, INT_MIN, INT_MAX, 1) before
// falling back to pseudo-random values of both signs.
fn lane_value(rng: &mut Lcg, index: usize) -> u32 {
    match index % 8 {
        0 => 0,
        1 => 0xffff_ffff,
        2 => 0x8000_0000,
        3 => 0x7fff_ffff,
        4 => 1,
        _ => rng.next(),
    }
}

pub fn gen_case(index: u32) -> ReluCase {
    let entries = model::bank_entries();
    // (rob_id, bank, iter, stride, cols, group, group_base)
    let (rob_id, bank, iter, stride, cols, group, group_base) = match index {
        0 => (1, 0, 16, entries, 1, 0, 0),
        1 => (2, 3, 32, entries / 2, 1, 0, 0),
        2 => (3, 5, 16, 16, 1, 0, 0),
        3 => (4, 7, 48, 256, 2, 1, 2),
        4 => (5, 9, 64, 64, 4, 3, 0),
        5 => (6, 2, entries, entries, 3, 2, 1),
        _ => panic!("relu: unsupported case {index}"),
    };
    model::validate(entries, iter, u64::from(stride), group, cols);
    let rows = model::rows(entries, iter, stride);
    let mut rng = Lcg(0x5eed_0000 + u64::from(index));
    let mut lane = 0usize;
    let src: Vec<u128> = rows
        .iter()
        .map(|_| {
            let mut word = 0u128;
            for l in 0..4 {
                word |= u128::from(lane_value(&mut rng, lane)) << (32 * l);
                lane += 1;
            }
            word
        })
        .collect();
    let dst = src.iter().map(|&w| model::relu_word(w)).collect();
    ReluCase {
        cmd: ReluCmd {
            iter,
            stride,
            bank,
            group_base,
            group,
            cols,
            rob_id,
            num_words: rows.len() as u32,
        },
        rows,
        src,
        dst,
    }
}
