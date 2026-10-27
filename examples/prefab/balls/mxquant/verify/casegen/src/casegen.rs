// MXQUANT reference model, the contract shared with the BEMU model
// (emu/src/quant.rs) and MxquantBall:
//
// op1 bank: count FP32 activations, four per row; block g of 32 is rows
//           8g..8g+7. Activations must be finite.
// wr bank:  E4M3 code of activation i in byte i (block g in rows 2g, 2g+1),
//           the E8M0 scale of block g in byte count + g.
// The block exponent is max(exponent(max |x|) - 8, -127) (0 for an all-zero
// block); codes round to nearest even and clip to 126 (448).
// rs2 and op2 bank are zero; count is a positive multiple of 32 with
// count <= 4 * bank_entries and count + count / 32 <= 16 * bank_entries.

pub const ROW_BYTES: usize = 16;
pub const BLOCK: usize = 32;
pub const NUM_CASES: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MxquantCmd {
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
    pub cmd: MxquantCmd,
    pub preload: Vec<Preload>,
    pub reads: Vec<u32>,
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

fn rounded_shift(value: u32, shift: u32) -> u32 {
    if shift > 31 {
        return 0;
    }
    let half = 1 << (shift - 1);
    (value + half - 1 + ((value >> shift) & 1)) >> shift
}

fn encode(bits: u32, block_exponent: i32) -> u8 {
    let sign = ((bits >> 24) & 128) as u8;
    if bits & 0x7fff_ffff == 0 {
        return sign;
    }
    let mut exponent = ((bits >> 23) & 255) as i32;
    let mut fraction = bits & 0x7f_ffff;
    if exponent == 0 {
        let shift = fraction.leading_zeros() - 8;
        fraction = (fraction << shift) & 0x7f_ffff;
        exponent = 1 - shift as i32;
    }
    exponent -= block_exponent + 120;
    let code = if exponent <= 0 {
        rounded_shift(fraction | 0x80_0000, (21 - exponent) as u32)
    } else {
        exponent as u32 * 8 + rounded_shift(fraction, 20)
    };
    sign | code.min(126) as u8
}

fn step(bank_entries: u32, rob_id: u32, banks: [u32; 2], groups: [u32; 2], input: &[u32]) -> Step {
    let count = input.len();
    let bytes = bank_entries as usize * ROW_BYTES;
    if count == 0 || count % BLOCK != 0 || count > bytes / 4 || count + count / BLOCK > bytes {
        panic!("mxquant: count must be a positive multiple of 32 within the bank");
    }
    if banks[0] == banks[1] {
        panic!("mxquant: input and output banks must differ");
    }
    if input.iter().any(|x| x & 0x7fff_ffff >= 0x7f80_0000) {
        panic!("mxquant: non-finite activation");
    }
    let mut preload = Vec::new();
    let mut reads = Vec::new();
    for (row, values) in input.chunks(4).enumerate() {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        preload.push(Preload { bank: banks[0], group: groups[0], addr: row as u32, data: word(&bytes) });
        reads.push(row as u32);
    }
    let mut writes = Vec::new();
    for (block, values) in input.chunks(BLOCK).enumerate() {
        let maximum = values.iter().map(|x| x & 0x7fff_ffff).max().unwrap();
        let exponent = if maximum == 0 { 0 } else { ((maximum >> 23) as i32 - 135).max(-127) };
        let codes: Vec<u8> = values.iter().map(|x| encode(*x, exponent)).collect();
        writes.push(Write { addr: (block * 2) as u32, mask: 0xffff, data: word(&codes[..ROW_BYTES]) });
        writes.push(Write { addr: (block * 2 + 1) as u32, mask: 0xffff, data: word(&codes[ROW_BYTES..]) });
        let byte = count + block;
        let lane = byte % ROW_BYTES;
        writes.push(Write {
            addr: (byte / ROW_BYTES) as u32,
            mask: 1 << lane,
            data: u128::from((exponent + 127) as u8) << (lane * 8),
        });
    }
    Step {
        cmd: MxquantCmd {
            rob_id,
            iter: count as u32,
            rs2_lo: 0,
            rs2_hi: 0,
            op1_bank: banks[0],
            op2_bank: 0,
            wr_bank: banks[1],
            op1_group: groups[0],
            op2_group: 0,
            wr_group: groups[1],
            op1_col: 1,
            op2_col: 0,
            wr_col: 1,
        },
        preload,
        reads,
        writes,
    }
}

// Activations of mxquant_values_test: test 0..3 are signed zeros, edges,
// rounding ties and subnormals; later tests sweep the exponent range.
fn values_test(rng: &mut u32, test: u32) -> Vec<u32> {
    let edges = [0, 0x8000_0000, 1, 0x8000_0001, 0x007f_ffff, 0x0080_0000, 0x3f88_0000, 0x3f98_0000, 0x43e0_0000, 0x7f7f_ffff, 0xff7f_ffff];
    let subnormals = [0, 1, 0x1000, 0x2000, 0x3000, 0x7f_ffff, 0x8000_0001, 0x807f_ffff];
    (0..BLOCK as u32)
        .map(|i| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            let random = *rng;
            match test {
                0 => if i & 1 != 0 { 0x8000_0000 } else { 0 },
                1 => edges[(i % 11) as usize],
                2 => if i == 0 { 0x43e0_0000 } else if i & 1 != 0 { 0x3f88_0000 } else { 0x3f98_0000 },
                3 => subnormals[(i % 8) as usize],
                _ => (random & 0x807f_ffff) | ((((test - 4) * 4 + (random >> 23 & 3)) % 255) << 23),
            }
        })
        .collect()
}

pub fn gen_case(index: u32, bank_entries: u32) -> Case {
    let mut rng = Rng(0x2545_f491 ^ index.wrapping_mul(0x9e37_79b9));
    let steps = match index {
        // Signed zeros, edges, rounding ties and subnormals, one block each.
        0 => {
            let mut state = 0x6d2b_79f5u32;
            let input: Vec<u32> = (0..4).flat_map(|test| values_test(&mut state, test)).collect();
            vec![step(bank_entries, 1, [3, 4], [0, 0], &input)]
        }
        // The exponent sweep of mxquant_values_test: 64 blocks, the full input
        // capacity of a 512-row bank.
        1 => {
            let mut state = 0x6d2b_79f5u32;
            let input: Vec<u32> = (0..68).flat_map(|test| values_test(&mut state, test)).skip(4 * BLOCK).collect();
            vec![step(bank_entries, 2, [3, 4], [1, 2], &input)]
        }
        // One block, random magnitudes around one, highest group bases.
        2 => {
            let input: Vec<u32> = (0..BLOCK).map(|_| (rng.next() & 0x807f_ffff) | ((120 + rng.next() % 16) << 23)).collect();
            vec![step(bank_entries, 15, [1022, 1023], [31, 31], &input)]
        }
        // Random exponents over the whole finite range, 3 / 4 of the input capacity.
        3 => {
            let count = (bank_entries * 3) as usize;
            let input: Vec<u32> = (0..count).map(|_| (rng.next() & 0x807f_ffff) | ((rng.next() % 255) << 23)).collect();
            vec![step(bank_entries, 7, [0, 1], [5, 6], &input)]
        }
        // Blocks whose maximum is subnormal or tiny (block exponent clamps to -127).
        4 => {
            let input: Vec<u32> = (0..2 * BLOCK)
                .map(|i| if i < BLOCK { rng.next() & 0x803f_ffff } else { (rng.next() & 0x807f_ffff) | ((rng.next() % 3) << 23) })
                .collect();
            vec![step(bank_entries, 4, [9, 10], [3, 4], &input)]
        }
        // Back-to-back commands without reset.
        5 => (0..3u32)
            .map(|i| {
                let count = BLOCK * (1 + (rng.next() % 5) as usize);
                let input: Vec<u32> = (0..count).map(|_| (rng.next() & 0x807f_ffff) | ((100 + rng.next() % 60) << 23)).collect();
                step(bank_entries, 8 + i, [11 + i, 20 + i], [i, i + 7], &input)
            })
            .collect(),
        _ => panic!("mxquant: unsupported case {index}"),
    };
    Case { steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_build() {
        for index in 0..NUM_CASES {
            gen_case(index, 512);
        }
    }

    // Golden bytes checked by mxquant_values_test.
    #[test]
    fn values_test_golden() {
        let mut state = 0x6d2b_79f5u32;
        let block = |state: &mut u32, test| {
            let input = values_test(state, test);
            let maximum = input.iter().map(|x| x & 0x7fff_ffff).max().unwrap();
            let exponent = if maximum == 0 { 0 } else { ((maximum >> 23) as i32 - 135).max(-127) };
            let codes: Vec<u8> = input.iter().map(|x| encode(*x, exponent)).collect();
            (codes, (exponent + 127) as u8)
        };
        let (codes, scale) = block(&mut state, 0);
        assert_eq!((codes[0], codes[1], scale), (0, 128, 127));
        let (codes, scale) = block(&mut state, 1);
        assert_eq!((codes[9], codes[10], scale), (126, 254, 246));
        let (codes, _) = block(&mut state, 2);
        assert_eq!((codes[0], codes[1], codes[2]), (126, 56, 58));
    }
}
