pub const ROW_BYTES: usize = 16;
pub const MAX_ROWS: usize = 64;
pub const NUM_CASES: u32 = 3;
pub const GATE_ROW: u32 = 1;
pub const RATIO: f32 = 0.25;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8MulCmd {
    pub bid: u32,
    pub iter: u32,
    pub gate_bank: u32,
    pub input_bank: u32,
    pub output_bank: u32,
    pub op1_col: u32,
    pub op2_col: u32,
    pub wr_col: u32,
    pub gate_row: u32,
    pub bank_entries: u32,
    pub rob_id: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub num_gate_words: u32,
    pub num_input_words: u32,
    pub num_dst_words: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int8MulCase {
    pub cmd: Int8MulCmd,
    pub gate_words: [u128; MAX_ROWS],
    pub input_words: [u128; MAX_ROWS],
    pub dst_words: [u128; MAX_ROWS],
}

impl Int8MulCase {
    pub fn gate_lo(&self, index: usize) -> u64 {
        self.gate_words[index] as u64
    }

    pub fn gate_hi(&self, index: usize) -> u64 {
        (self.gate_words[index] >> 64) as u64
    }

    pub fn input_lo(&self, index: usize) -> u64 {
        self.input_words[index] as u64
    }

    pub fn input_hi(&self, index: usize) -> u64 {
        (self.input_words[index] >> 64) as u64
    }

    pub fn dst_lo(&self, index: usize) -> u64 {
        self.dst_words[index] as u64
    }

    pub fn dst_hi(&self, index: usize) -> u64 {
        (self.dst_words[index] >> 64) as u64
    }
}

pub fn gen_case(index: u32, bid: u32, bank_entries: u32) -> Int8MulCase {
    if bank_entries == 0 {
        panic!("int8mul: bank entries must be positive");
    }
    if index >= NUM_CASES {
        panic!("int8mul: unsupported directed case {index}");
    }
    match index {
        0 => directed(bid, 2, 1, GATE_ROW, bank_entries),
        1 => directed(bid, 3, 4, GATE_ROW, bank_entries),
        2 => directed(bid, 4, 1, bank_entries - 1, bank_entries),
        _ => panic!("int8mul: unsupported directed case {index}"),
    }
}

fn compute(gate: i8, input: i8, ratio: f32) -> i8 {
    let value = (gate as f32) * (input as f32) * ratio;
    value.round_ties_even().clamp(-128.0, 127.0) as i8
}

fn positive_finite(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn ctest_gate() -> u128 {
    let mut row = [0u8; ROW_BYTES];
    for lane in 0..ROW_BYTES {
        row[lane] = (lane as i8 - 8) as u8;
    }
    pack_bytes(&row)[0]
}

fn ctest_input(rows: usize) -> Vec<u128> {
    let count = rows * ROW_BYTES;
    let bytes: Vec<u8> = (0..count)
        .map(|i| ((i as i32 * 37 & 255) - 128) as i8 as u8)
        .collect();
    pack_bytes(&bytes)
}

fn pack_bytes(bytes: &[u8]) -> Vec<u128> {
    if bytes.len() % ROW_BYTES != 0 {
        panic!(
            "int8mul: {} bytes is not a multiple of {ROW_BYTES}",
            bytes.len()
        );
    }
    bytes
        .chunks(ROW_BYTES)
        .map(|chunk| {
            let mut word = 0u128;
            for (lane, byte) in chunk.iter().enumerate() {
                word |= u128::from(*byte) << (lane * 8);
            }
            word
        })
        .collect()
}

fn copy_words<const N: usize>(src: &[u128]) -> [u128; N] {
    if src.len() > N {
        panic!("int8mul: {} words exceed buffer {N}", src.len());
    }
    let mut out = [0u128; N];
    out[..src.len()].copy_from_slice(src);
    out
}

fn encode_rs1(gate: u32, input: u32, output: u32, iter: u32) -> u64 {
    u64::from(gate) | (u64::from(input) << 10) | (u64::from(output) << 20) | (u64::from(iter) << 30)
}

fn encode_rs2(ratio: f32, gate_row: u32) -> u64 {
    u64::from(ratio.to_bits()) | (u64::from(gate_row) << 32)
}

fn split_rs(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

pub(crate) fn validate(
    gate_bank: u32,
    input_bank: u32,
    output_bank: u32,
    op1_col: u32,
    op2_col: u32,
    wr_col: u32,
    iter: u32,
    gate_row: u32,
    rs1: u64,
    rs2: u64,
    ratio: f32,
    bank_entries: u32,
) {
    if bank_entries == 0 {
        panic!("int8mul: bank entries must be positive");
    }
    if iter == 0 || iter > bank_entries {
        panic!("int8mul: iter must fit one physical bank");
    }
    if gate_row >= bank_entries {
        panic!("int8mul: gate row must fit one physical bank");
    }
    if op1_col != 1 || op2_col != 1 || wr_col != 1 {
        panic!("int8mul: bank groups must match");
    }
    if gate_bank == input_bank || gate_bank == output_bank || input_bank == output_bank {
        panic!("int8mul: banks must be distinct");
    }
    if !positive_finite(ratio) {
        panic!("int8mul: ratio must be finite and positive");
    }
    let gate_row_bits = u32::BITS - (bank_entries - 1).leading_zeros();
    if rs2 >> (32 + gate_row_bits) != 0 {
        panic!("int8mul: reserved rs2 bits must be zero");
    }
    if (rs1 & 0x3ff) != u64::from(gate_bank)
        || ((rs1 >> 10) & 0x3ff) != u64::from(input_bank)
        || ((rs1 >> 20) & 0x3ff) != u64::from(output_bank)
    {
        panic!("int8mul: rs1 bank fields mismatch");
    }
    if ((rs1 >> 30) & 0x3_ffff_ffff) != u64::from(iter) {
        panic!("int8mul: rs1 iter mismatch");
    }
    if rs2 != encode_rs2(ratio, gate_row) {
        panic!("int8mul: rs2 ratio/gate_row fields mismatch");
    }
}

fn dst_words(gate_word: u128, input: &[u128], iter: u32, ratio: f32) -> Vec<u128> {
    let mut out = Vec::with_capacity(iter as usize);
    for row in 0..iter as usize {
        let mut packed = 0u128;
        for lane in 0..ROW_BYTES {
            let g = ((gate_word >> (lane * 8)) & 0xff) as u8 as i8;
            let inp = ((input[row] >> (lane * 8)) & 0xff) as u8 as i8;
            let q = compute(g, inp, ratio);
            packed |= u128::from(q as u8) << (lane * 8);
        }
        out.push(packed);
    }
    out
}

fn directed(bid: u32, rob_id: u32, iter: u32, gate_row: u32, bank_entries: u32) -> Int8MulCase {
    let gate_bank = 0;
    let input_bank = 1;
    let output_bank = 2;
    let op1_col = 1;
    let op2_col = 1;
    let wr_col = 1;
    let ratio = RATIO;
    let gate = ctest_gate();
    let input = ctest_input(iter as usize);
    let rs1 = encode_rs1(gate_bank, input_bank, output_bank, iter);
    let rs2 = encode_rs2(ratio, gate_row);
    validate(
        gate_bank,
        input_bank,
        output_bank,
        op1_col,
        op2_col,
        wr_col,
        iter,
        gate_row,
        rs1,
        rs2,
        ratio,
        bank_entries,
    );
    if input.len() != iter as usize {
        panic!("int8mul: input rows {} != iter {iter}", input.len());
    }
    let dst = dst_words(gate, &input, iter, ratio);
    let (rs1_lo, rs1_hi) = split_rs(rs1);
    let (rs2_lo, rs2_hi) = split_rs(rs2);
    Int8MulCase {
        cmd: Int8MulCmd {
            bid,
            iter,
            gate_bank,
            input_bank,
            output_bank,
            op1_col,
            op2_col,
            wr_col,
            gate_row,
            bank_entries,
            rob_id,
            rs1_lo,
            rs1_hi,
            rs2_lo,
            rs2_hi,
            num_gate_words: 1,
            num_input_words: input.len() as u32,
            num_dst_words: dst.len() as u32,
        },
        gate_words: copy_words(&[gate]),
        input_words: copy_words(&input),
        dst_words: copy_words(&dst),
    }
}
