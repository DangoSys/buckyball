// MXFP2INT reference model (MXFP4 to INT8), the contract shared with the
// BEMU model (emu/src/55_mxfp2int.rs) and Mxfp2IntBall:
//
// op1 bank: block b in row b, element e in byte e / 2, low nibble first; the
//           element value is raw - 8.
// op2 bank: E8M0 scale of block b in byte b % 16 of row b / 16.
// wr bank:  elements 0..15 of block b in row 2b, 16..31 in row 2b + 1,
//           saturate(value * 2^(scale - 127)); the right shift rounds toward
//           minus infinity.
// rs2 is reserved and zero; 1 <= iter <= bank_entries / 2.

pub const ROW_BYTES: usize = 16;
pub const ELEMENTS: usize = 32;
pub const NUM_CASES: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mxfp2IntCmd {
    pub rob_id: u32,
    pub iter: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub op1_bank: u32,
    pub op2_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub op2_col: u32,
    pub wr_col: u32,
}

#[derive(Clone, Debug)]
pub struct Preload {
    pub bank: u32,
    pub group: u32,
    pub addr: u32,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Write {
    pub addr: u32,
    pub mask: u16,
    pub data: u128,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub cmd: Mxfp2IntCmd,
    pub preload: Vec<Preload>,
    pub reads: [Vec<u32>; 2],
    pub writes: Vec<Write>,
}

#[derive(Clone, Debug)]
pub struct Case {
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
}

fn word(bytes: &[u8]) -> u128 {
    let mut out = 0u128;
    for (lane, byte) in bytes.iter().enumerate() {
        out |= u128::from(*byte) << (lane * 8);
    }
    out
}

fn dequant(raw: u8, scale: u8) -> u8 {
    let value = raw as i32 - 8;
    let shift = scale as i32 - 127;
    let scaled = if shift >= 0 {
        // |value| <= 8, so shifts past 7 saturate.
        (value as i64) << shift.min(8)
    } else {
        (value >> (-shift).min(31)) as i64
    };
    scaled.clamp(-128, 127) as i8 as u8
}

// `data` holds iter rows of packed nibbles, `scales` one byte per block.
fn step(
    bank_entries: u32,
    rob_id: u32,
    banks: [u32; 3],
    groups: [u32; 3],
    data: &[u8],
    scales: &[u8],
) -> Step {
    let blocks = scales.len();
    if blocks == 0 || blocks * 2 > bank_entries as usize || data.len() != blocks * ROW_BYTES {
        panic!("mxfp2int: iter must be 1..half the bank depth");
    }
    if banks[0] == banks[1] || banks[0] == banks[2] || banks[1] == banks[2] {
        panic!("mxfp2int: banks must differ");
    }
    let mut scale_rows = vec![0u8; blocks.div_ceil(ROW_BYTES) * ROW_BYTES];
    scale_rows[..blocks].copy_from_slice(scales);
    let mut preload = Vec::new();
    for (row, bytes) in data.chunks(ROW_BYTES).enumerate() {
        preload.push(Preload { bank: banks[0], group: groups[0], addr: row as u32, data: word(bytes) });
    }
    for (row, bytes) in scale_rows.chunks(ROW_BYTES).enumerate() {
        preload.push(Preload { bank: banks[1], group: groups[1], addr: row as u32, data: word(bytes) });
    }
    let mut reads = [Vec::new(), Vec::new()];
    let mut writes = Vec::new();
    for block in 0..blocks {
        reads[0].push(block as u32);
        reads[1].push((block / ROW_BYTES) as u32);
        let row = &data[block * ROW_BYTES..(block + 1) * ROW_BYTES];
        let mut result = [0u8; ELEMENTS];
        for (element, byte) in result.iter_mut().enumerate() {
            *byte = dequant((row[element / 2] >> ((element % 2) * 4)) & 0x0f, scales[block]);
        }
        writes.push(Write { addr: (block * 2) as u32, mask: 0xffff, data: word(&result[..ROW_BYTES]) });
        writes.push(Write { addr: (block * 2 + 1) as u32, mask: 0xffff, data: word(&result[ROW_BYTES..]) });
    }
    Step {
        cmd: Mxfp2IntCmd {
            rob_id,
            iter: blocks as u32,
            rs2_lo: 0,
            rs2_hi: 0,
            op1_bank: banks[0],
            op2_bank: banks[1],
            wr_bank: banks[2],
            op1_group: groups[0],
            op2_group: groups[1],
            wr_group: groups[2],
            op1_col: 1,
            op2_col: 1,
            wr_col: 1,
        },
        preload,
        reads,
        writes,
    }
}

fn pattern_data(blocks: usize) -> Vec<u8> {
    let raw = |block: usize, elem: usize| ((elem * 7 + block * 3) & 0xf) as u8;
    let mut data = vec![0u8; blocks * ROW_BYTES];
    for block in 0..blocks {
        for elem in (0..ELEMENTS).step_by(2) {
            data[block * ROW_BYTES + elem / 2] = (raw(block, elem + 1) << 4) | raw(block, elem);
        }
    }
    data
}

fn random_data(rng: &mut Rng, blocks: usize) -> Vec<u8> {
    (0..blocks * ROW_BYTES).map(|_| rng.next() as u8).collect()
}

pub fn gen_case(index: u32, bank_entries: u32) -> Case {
    let mut rng = Rng(0x6d2b_79f5 ^ index.wrapping_mul(0x9e37_79b9));
    let steps = match index {
        // Two blocks at the identity scale (mxfp4_e8m0_two_block_test).
        0 => vec![step(bank_entries, 1, [1, 2, 3], [0, 0, 0], &pattern_data(2), &[127, 127])],
        // Scales across right shifts, saturating left shifts and shifts past 31,
        // spanning two scale rows (mxfp4_e8m0_scale_range_test).
        1 => {
            let scales = [0, 1, 90, 120, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 150, 158, 159, 200, 255];
            vec![step(bank_entries, 2, [1, 2, 3], [0, 0, 0], &pattern_data(scales.len()), &scales)]
        }
        // Random nibbles and scales near the identity, non-zero group bases.
        2 => {
            let scales: Vec<u8> = (0..100).map(|_| 120 + (rng.next() % 16) as u8).collect();
            let data = random_data(&mut rng, scales.len());
            vec![step(bank_entries, 5, [4, 7, 9], [3, 11, 6], &data, &scales)]
        }
        // One block, highest banks and groups.
        3 => vec![step(bank_entries, 15, [1021, 1022, 1023], [15, 15, 15], &random_data(&mut rng, 1), &[129])],
        // Full capacity: iter = bank_entries / 2.
        4 => {
            let blocks = (bank_entries / 2) as usize;
            let scales: Vec<u8> = (0..blocks).map(|_| rng.next() as u8).collect();
            let data = random_data(&mut rng, blocks);
            vec![step(bank_entries, 9, [0, 512, 768], [1, 2, 4], &data, &scales)]
        }
        // Back-to-back commands without reset.
        5 => (0..3)
            .map(|i| {
                let blocks = 1 + (rng.next() % 18) as usize;
                let scales: Vec<u8> = (0..blocks).map(|_| 118 + (rng.next() % 20) as u8).collect();
                let data = random_data(&mut rng, blocks);
                step(bank_entries, 6 + i, [8 + i, 16 + i, 24 + i], [i, i + 4, i + 8], &data, &scales)
            })
            .collect(),
        _ => panic!("mxfp2int: unsupported case {index}"),
    };
    Case { steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_build() {
        for index in 0..NUM_CASES {
            gen_case(index, 1024);
        }
    }

    // Golden values of mxfp4_e8m0_scale_range_test.
    #[test]
    fn scale_range_golden() {
        assert_eq!(dequant(15, 127) as i8, 7);
        assert_eq!(dequant(0, 127) as i8, -8);
        assert_eq!(dequant(9, 0) as i8, 0);
        assert_eq!(dequant(7, 0) as i8, -1);
        assert_eq!(dequant(15, 255) as i8, 127);
        assert_eq!(dequant(0, 255) as i8, -128);
        assert_eq!(dequant(1, 126) as i8, -4);
        assert_eq!(dequant(15, 130) as i8, 56);
    }
}
