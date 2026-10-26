use crate::model;

pub const BIAS_BANK: u32 = 0;
pub const C_BANK: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatrixCmd {
    pub kind: u32,
    pub rob_id: u32,
    pub op1_words: u32,
    pub op2_words: u32,
    pub rs1_lo: u32,
    pub rs1_hi: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub op1_group: u32,
    pub op2_group: u32,
    pub wr_group: u32,
    pub k_groups: u32,
    pub wr_col: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixCase {
    pub commands: Vec<MatrixCmd>,
    pub bias: Vec<u8>,
    // a[block] / b[block] hold every K group of the block, group-major.
    pub a: Vec<Vec<u8>>,
    pub b: Vec<Vec<u8>>,
    pub writes: Vec<model::WriteExp>,
}

// Random cases also place every operand at a nonzero group base of its bank.
pub fn gen_case(seed: u32, index: u32, tile_rows: usize) -> MatrixCase {
    match index {
        0 => build_case(seed, 16, 1, tile_rows, false),
        1 => build_case(seed, 16, 2, tile_rows, false),
        2 => build_case(seed ^ index, 16, 1, tile_rows, false),
        11 => three_k_groups(seed, tile_rows),
        12 => c_crosses_groups(seed, tile_rows),
        _ => {
            let mut rng = Rng::new(seed, index);
            let blocks = 1 + (rng.next() % 2) as usize;
            build_case(rng.next(), 16, blocks, tile_rows, true)
        }
    }
}

fn build_case(
    seed: u32,
    rows: usize,
    blocks: usize,
    tile_rows: usize,
    grouped: bool,
) -> MatrixCase {
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
    let group = |rng: &mut Rng| if grouped { 1 + rng.next() % 3 } else { 0 };
    let mut commands = Vec::with_capacity(blocks + 1);
    let bias_group = group(&mut rng);
    commands.push(MatrixCmd {
        kind: 0,
        rob_id: 0,
        op1_group: bias_group,
        op1_words: 4,
        k_groups: 1,
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
        let rs2 = model::encode_rs2(rows, block == 0, block + 1 == blocks, 0);
        let op1_group = group(&mut rng);
        let op2_group = group(&mut rng);
        let wr_group = group(&mut rng);
        commands.push(MatrixCmd {
            kind: 1,
            rob_id: (block + 1) as u32,
            op1_group,
            op2_group,
            wr_group,
            op1_words: rows as u32,
            op2_words: (model::CORE * rows.div_ceil(tile_rows)) as u32,
            k_groups: 1,
            wr_col: 1,
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
        writes: model::emit_writes(&result, 0),
    }
}

fn bias_command(op1_group: u32) -> MatrixCmd {
    let rs1 = model::encode_rs1(BIAS_BANK, 0, 0, 4);
    MatrixCmd {
        kind: 0,
        op1_group,
        op1_words: 4,
        k_groups: 1,
        rs1_lo: rs1 as u32,
        rs1_hi: (rs1 >> 32) as u32,
        ..MatrixCmd::default()
    }
}

fn bias_rows(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

// M=16, K=48 in one command: A and B have three groups, group g holding
// K block g (16 per group), at nonzero group bases.
fn three_k_groups(seed: u32, tile_rows: usize) -> MatrixCase {
    let mut rng = Rng::new(seed, 11);
    let bias: Vec<i32> = (0..model::CORE).map(|_| rng.value() as i32).collect();
    let a_values: Vec<i8> = (0..16 * 48).map(|_| rng.value()).collect();
    let b_values: Vec<i8> = (0..48 * model::CORE).map(|_| rng.value()).collect();
    // Group g of A is A[:, 16g..16g+16] row-major; group g of B is rows 16g...
    let mut a = Vec::new();
    for group in 0..3 {
        for row in 0..16 {
            for k in 0..16 {
                a.push(a_values[row * 48 + group * 16 + k] as u8);
            }
        }
    }
    let b: Vec<u8> = b_values.iter().map(|value| *value as u8).collect();
    let mut result = vec![bias.clone(); 16];
    for row in 0..16 {
        for column in 0..model::CORE {
            for k in 0..48 {
                result[row][column] = result[row][column].wrapping_add(
                    a_values[row * 48 + k] as i32 * b_values[k * model::CORE + column] as i32,
                );
            }
        }
    }
    let rs1 = model::encode_rs1(1, 2, C_BANK, 16);
    let rs2 = model::encode_rs2(16, true, true, 0);
    MatrixCase {
        commands: vec![
            bias_command(0),
            MatrixCmd {
                kind: 1,
                rob_id: 1,
                op1_group: 1,
                op2_group: 2,
                wr_group: 3,
                op1_words: 16,
                op2_words: (model::CORE * 16_usize.div_ceil(tile_rows)) as u32,
                k_groups: 3,
                wr_col: 1,
                rs1_lo: rs1 as u32,
                rs1_hi: (rs1 >> 32) as u32,
                rs2_lo: rs2 as u32,
                rs2_hi: (rs2 >> 32) as u32,
            },
        ],
        bias: bias_rows(&bias),
        a: vec![a],
        b: vec![b],
        writes: model::emit_writes(&result, 0),
    }
}

// M=128, K=32 (two K groups), output_base 32: C lines 32..543 continue from
// C group 0 into group 1 on 512-line banks.
fn c_crosses_groups(seed: u32, tile_rows: usize) -> MatrixCase {
    let mut rng = Rng::new(seed, 12);
    let bias: Vec<i32> = (0..model::CORE).map(|_| rng.value() as i32).collect();
    let a_values: Vec<i8> = (0..128 * 32).map(|_| rng.value()).collect();
    let b_values: Vec<i8> = (0..32 * model::CORE).map(|_| rng.value()).collect();
    let mut a = Vec::new();
    for group in 0..2 {
        for row in 0..128 {
            for k in 0..16 {
                a.push(a_values[row * 32 + group * 16 + k] as u8);
            }
        }
    }
    let b: Vec<u8> = b_values.iter().map(|value| *value as u8).collect();
    let mut result = vec![bias.clone(); 128];
    for row in 0..128 {
        for column in 0..model::CORE {
            for k in 0..32 {
                result[row][column] = result[row][column].wrapping_add(
                    a_values[row * 32 + k] as i32 * b_values[k * model::CORE + column] as i32,
                );
            }
        }
    }
    let rs1 = model::encode_rs1(1, 2, C_BANK, 16);
    let rs2 = model::encode_rs2(128, true, true, 32);
    MatrixCase {
        commands: vec![
            bias_command(0),
            MatrixCmd {
                kind: 1,
                rob_id: 1,
                op1_group: 0,
                op2_group: 0,
                wr_group: 0,
                op1_words: 128,
                op2_words: (model::CORE * 128_usize.div_ceil(tile_rows)) as u32,
                k_groups: 2,
                wr_col: 2,
                rs1_lo: rs1 as u32,
                rs1_hi: (rs1 >> 32) as u32,
                rs2_lo: rs2 as u32,
                rs2_hi: (rs2 >> 32) as u32,
            },
        ],
        bias: bias_rows(&bias),
        a: vec![a],
        b: vec![b],
        writes: model::emit_writes(&result, 32),
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
