use crate::model;

pub const BIAS_BANK: u32 = 0;
pub const C_BANK: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatrixCmd {
    pub kind: u32,
    pub bid: u32,
    pub rob_id: u32,
    pub op1_words: u32,
    pub op2_words: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixCase {
    pub commands: Vec<MatrixCmd>,
    pub bias: Vec<u8>,
    pub a: Vec<Vec<u8>>,
    pub b: Vec<Vec<u8>>,
    pub writes: Vec<model::WriteExp>,
}

pub fn gen_case(seed: u32, index: u32, bid: u32, tile_rows: usize) -> MatrixCase {
    match index {
        0 => build_case(seed, bid, 16, 1, tile_rows),
        1 => build_case(seed, bid, 16, 2, tile_rows),
        2 => build_case(seed ^ index, bid, 16, 1, tile_rows),
        _ => {
            let mut rng = Rng::new(seed, index);
            let blocks = 1 + (rng.next() % 2) as usize;
            build_case(rng.next(), bid, 16, blocks, tile_rows)
        }
    }
}

fn build_case(seed: u32, bid: u32, rows: usize, blocks: usize, tile_rows: usize) -> MatrixCase {
    assert!(matches!(rows, 16 | 32 | 64));
    assert!(matches!(blocks, 1 | 2));
    assert!(rows * model::RESULT_WORDS <= model::BANK_DEPTH);
    let mut rng = Rng::new(seed, rows as u32 ^ blocks as u32);
    let bias_values: Vec<i32> = (0..model::CORE).map(|_| rng.value() as i32).collect();
    let mut bias = Vec::with_capacity(4 * model::ROW_BYTES);
    for chunk in bias_values.chunks(4) {
        for value in chunk {
            bias.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut a = Vec::with_capacity(blocks);
    let mut b = Vec::with_capacity(blocks);
    let mut result = vec![bias_values.clone(); rows];
    let mut commands = Vec::with_capacity(blocks + 1);
    commands.push(MatrixCmd {
        kind: 0,
        bid,
        rob_id: 0,
        op1_words: 4,
        rs1_lo: model::encode_rs1(BIAS_BANK, 0, 0, 4) as u32,
        rs1_hi: (model::encode_rs1(BIAS_BANK, 0, 0, 4) >> 32) as u32,
        ..MatrixCmd::default()
    });

    for block in 0..blocks {
        let a_bank = 1 + 2 * block as u32;
        let b_bank = 2 + 2 * block as u32;
        let a_values: Vec<i8> = (0..rows * model::CORE).map(|_| rng.value()).collect();
        let b_values: Vec<i8> = (0..model::CORE * model::CORE)
            .map(|_| rng.value())
            .collect();
        let a_packed = model::pack_a(&a_values, rows);
        let b_packed = model::pack_b(&b_values);
        a.push(a_packed);
        b.push(b_packed);
        for row in 0..rows {
            for column in 0..model::CORE {
                for k in 0..model::CORE {
                    result[row][column] = result[row][column].wrapping_add(
                        a_values[row * model::CORE + k] as i32
                            * b_values[k * model::CORE + column] as i32,
                    );
                }
            }
        }
        let rs1 = model::encode_rs1(a_bank, b_bank, C_BANK, model::CORE as u64);
        let rs2 = model::encode_rs2(rows, block == 0, block + 1 == blocks);
        commands.push(MatrixCmd {
            kind: 1,
            bid,
            rob_id: (block + 1) as u32,
            op1_words: rows as u32,
            op2_words: (model::CORE * rows.div_ceil(tile_rows)) as u32,
            rs1_lo: rs1 as u32,
            rs1_hi: (rs1 >> 32) as u32,
            rs2_lo: rs2 as u32,
            rs2_hi: (rs2 >> 32) as u32,
        });
    }

    MatrixCase {
        commands,
        bias,
        a,
        b,
        writes: model::emit_writes(&result),
    }
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u32, salt: u32) -> Self {
        Self {
            state: u64::from(seed) << 32 | u64::from(salt),
        }
    }
    fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }
    fn value(&mut self) -> i8 {
        (self.next() % 17) as i8 - 8
    }
}
