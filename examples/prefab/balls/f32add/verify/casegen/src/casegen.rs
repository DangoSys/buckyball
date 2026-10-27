use crate::model;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32AddCmd {
    pub rows: u32,
    pub a_bank: u32,
    pub b_bank: u32,
    pub c_bank: u32,
    pub a_group_base: u32,
    pub b_group: u32,
    pub c_group: u32,
    pub a_cols: u32,
    pub group: u32,
    pub first: u32,
    pub rob_id: u32,
}

#[derive(Clone, Debug)]
pub struct F32AddCase {
    pub cmd: F32AddCmd,
    pub a: Vec<u128>,
    pub b: Vec<u128>,
    pub c: Vec<u128>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

// (source, accumulator) pairs at the IEEE edges: signed zeros, infinities,
// quiet and signalling NaNs, overflow, subnormal results and ties.
const EDGES: [(u32, u32); 20] = [
    (0x0000_0000, 0x8000_0000), // +0 + -0 = +0
    (0x8000_0000, 0x8000_0000), // -0 + -0 = -0
    (0x7f80_0000, 0xff80_0000), // inf - inf = NaN
    (0x7f80_0000, 0x3f80_0000), // inf + 1
    (0xff80_0000, 0xff80_0000), // -inf + -inf
    (0x7fc0_1234, 0x3f80_0000), // quiet NaN payload
    (0x3f80_0000, 0x7f80_0001), // signalling NaN
    (0xffff_ffff, 0x0000_0000), // negative NaN
    (0x7f7f_ffff, 0x7f7f_ffff), // overflow to inf
    (0xff7f_ffff, 0xff7f_fffe), // overflow to -inf
    (0x0000_0001, 0x0000_0001), // subnormal sum
    (0x0000_0001, 0x8000_0001), // exact cancellation to +0
    (0x007f_ffff, 0x0000_0001), // subnormal to the smallest normal
    (0x0080_0000, 0x8000_0001), // normal to subnormal
    (0x3f80_0000, 0x3380_0000), // 1 + 2^-24: tie, stays even
    (0x3f80_0001, 0x3380_0000), // tie rounds up to even
    (0x3f80_0000, 0x33c0_0000), // above the tie
    (0x3f80_0000, 0xbf80_0000), // 1 - 1 = +0
    (0x4b80_0000, 0xbf80_0000), // 2^24 - 1, exact
    (0x7f00_0000, 0xfeff_ffff), // large cancellation
];

fn lane_pair(rng: &mut Lcg, index: usize, edges: bool) -> (u32, u32) {
    if edges {
        return EDGES[index % EDGES.len()];
    }
    match index % 4 {
        // Nearby exponents exercise alignment and rounding.
        0 | 1 => {
            let a = rng.next();
            let b = (a & 0xff80_0000) ^ (rng.next() & 0x807f_ffff) ^ (rng.next() & 0x0380_0000);
            (a, b)
        }
        _ => (rng.next(), rng.next()),
    }
}

fn pack(lanes: &[u32]) -> Vec<u128> {
    lanes
        .chunks(4)
        .map(|chunk| {
            let mut word = 0u128;
            for (lane, value) in chunk.iter().enumerate() {
                word |= u128::from(*value) << (32 * lane);
            }
            word
        })
        .collect()
}

pub fn gen_case(index: u32) -> F32AddCase {
    let entries = model::bank_entries();
    // (rob_id, banks, rows, a_cols, group, first, group bases, edge values)
    let (rob_id, banks, rows, a_cols, group, first, bases, edges) = match index {
        0 => (1, [0, 1, 2], 4, 1, 0, 0, [0, 0, 0], false),
        1 => (2, [3, 1, 0], 7, 1, 0, 1, [0, 0, 0], false),
        2 => (3, [2, 0, 3], 16, 3, 2, 0, [1, 2, 3], true),
        3 => (4, [33, 600, 1023], entries, 1, 0, 0, [0, 1, 0], false),
        4 => (5, [5, 6, 7], 9, 2, 1, 1, [2, 0, 1], true),
        _ => panic!("f32add: unsupported case {index}"),
    };
    model::validate(entries, banks[0], banks[1], banks[2], rows, group, first, a_cols);
    let mut rng = Lcg(0xf32a_dd00 + u64::from(index));
    let lanes = rows as usize * 4;
    let mut a_lanes = Vec::with_capacity(lanes);
    let mut b_lanes = Vec::with_capacity(lanes);
    for i in 0..lanes {
        let (s, acc) = lane_pair(&mut rng, i, edges);
        a_lanes.push(s);
        b_lanes.push(acc);
    }
    let a = pack(&a_lanes);
    let b = pack(&b_lanes);
    let c = a
        .iter()
        .zip(&b)
        .map(|(&s, &acc)| model::add_word(s, acc, first == 1))
        .collect();
    F32AddCase {
        cmd: F32AddCmd {
            rows,
            a_bank: banks[0],
            b_bank: banks[1],
            c_bank: banks[2],
            a_group_base: bases[0],
            b_group: bases[1],
            c_group: bases[2],
            a_cols,
            group,
            first,
            rob_id,
        },
        a,
        b,
        c,
    }
}
