use crate::timing::{Timing, EXECUTE, LOAD, STORE};
use bebop_bemu::riscv::{registers::CustomInstruction, Trap};
use bebop_bemu::{GuestAccess, Rocc};

/// DefaultGemmini: a 16 x 16 INT8 mesh, INT32 accumulators, FP32 scales.
pub const DIM: usize = 16;
pub const ADDR_LEN: u32 = 32;
pub const BANK_NUM: usize = 4;
pub const BANK_ROWS: usize = 4096;
pub const ACC_ROWS: usize = 1024;
pub const MAX_BLOCK_LEN: usize = 64 / DIM;
pub const MAX_BLOCK_LEN_ACC: usize = 64 / (DIM * 4);
pub const GARBAGE_ADDR: u32 = !0;
const NORM_STAT_IDS: usize = 4;
const NUM_COUNTERS: usize = 8;

/// Activations, as config and loop instructions encode them.
pub const NONE: u8 = 0;
pub const RELU: u8 = 1;
pub const LAYERNORM: u8 = 2;
pub const IGELU: u8 = 3;
pub const SOFTMAX: u8 = 4;

/// Normalization commands of mvout (sp_addr[28:26]).
const RESET: u64 = 0;
const SUM: u64 = 1;
const MEAN: u64 = 2;
const VARIANCE: u64 = 3;
const INV_STDDEV: u64 = 4;
const MAX: u64 = 5;
const SUM_EXP: u64 = 6;
const INV_SUM_EXP: u64 = 7;

/// The DMA, 16 bytes a beat, streams a beat a cycle each way. A load's data
/// arrives LOAD_LATENCY after its last request; a store completes when its
/// writes are acknowledged, STORE_LATENCY after its last request.
const BEAT_BYTES: u64 = 16;
const LOAD_LATENCY: u64 = 50;
const STORE_LATENCY: u64 = 190;
/// The mesh: results reach the scratchpad or the accumulator after the PEs
/// and the write-back (RTL: a dependent preload follows a compute by ~100 cycles).
const MESH_LATENCY: u64 = 87;

/// Gemmini on the RoCC port of its Rocket CPU. The functional semantics are
/// those of the Spike reference (libgemmini, the commit of thirdparty/gemmini);
/// the timing is that of `Timing`.
pub struct Gemmini {
    pub(crate) spad: Vec<[i8; DIM]>,
    pub(crate) accumulator: Vec<[i32; DIM]>,
    pe_state: [[i32; DIM]; DIM],

    // Execute.
    pub(crate) output_sp_addr: u32,
    preload_sp_addr: u32,
    preload_cols: u16,
    preload_rows: u16,
    output_cols: u16,
    output_rows: u16,
    weight_stationary: bool,
    sys_act: u8,
    sys_shift: u32,
    a_transpose: bool,
    b_transpose: bool,
    c_stride: u16,
    a_stride: u16,

    // Load and store.
    pub(crate) acc_act: u8,
    pub(crate) acc_shift: f32,
    load_strides: [u64; 3],
    load_block_strides: [u16; 3],
    load_shrunks: [bool; 3],
    load_scales: [f32; 3],
    pixels_per_rows: [u8; 3],
    store_stride: u64,
    pool_stride: u8,
    pool_size: u8,
    pool_out_dim: u8,
    pool_porows: u8,
    pool_pocols: u8,
    pool_orows: u8,
    pool_ocols: u8,
    pool_lpad: u8,
    pool_upad: u8,

    // Normalization.
    igelu_qb: i32,
    igelu_qc: i32,
    qln2: i32,
    qln2_inv: i32,
    norm_stat_id: usize,
    norm_sum: [i32; NORM_STAT_IDS],
    norm_running_max: [i32; NORM_STAT_IDS],
    norm_max: [i32; NORM_STAT_IDS],
    norm_count: [i32; NORM_STAT_IDS],
    norm_mean: [i32; NORM_STAT_IDS],
    norm_inv_stddev: [f32; NORM_STAT_IDS],
    norm_inv_sum_exp: [f32; NORM_STAT_IDS],
    norm_reset: [bool; NORM_STAT_IDS],

    // The loop unrollers.
    pub(crate) loop_ws: [u64; 10],
    pub(crate) loop_conv: [u64; 12],
    pub(crate) conv_d_row: u32,
    pub(crate) conv_c_row: u32,
    /// When the unroller finished issuing each of the last two loops, older first.
    loop_issued: [u64; 2],

    // Performance counters: internal ones count the cycles Gemmini is busy,
    // external ones the loads holding the reservation station.
    counter_val: [u32; NUM_COUNTERS],
    counter_snapshot: [u32; NUM_COUNTERS],
    counter_external: [bool; NUM_COUNTERS],
    /// The busy cycles the counters have counted.
    counted: u64,
    snapshot_enable: bool,

    pub(crate) timing: Timing,
    buffer: Vec<u8>,
}

fn round_saturate_i8(value: f32) -> i8 {
    value.round_ties_even().clamp(-128.0, 127.0) as i8
}

fn acc_scale(value: i32, scale: f32) -> i8 {
    round_saturate_i8(value as f32 * scale)
}

fn sys_shift(value: i32, shift: u32) -> i8 {
    let shifted = if shift == 0 {
        value
    } else {
        let round = (value >> (shift - 1)) & 1;
        let sticky = shift > 1 && value & ((1 << (shift - 1)) - 1) != 0;
        (value >> shift) + (round & (i32::from(sticky) | ((value >> shift) & 1)))
    };
    shifted.clamp(-128, 127) as i8
}

fn activation(value: i8, act: u8) -> i8 {
    if act == RELU {
        value.max(0)
    } else {
        value
    }
}

fn igelu(q: i32, qb: i32, qc: i32) -> i32 {
    let sign = if q < 0 { -1 } else { 1 };
    let clipped = q.wrapping_abs().min(qb.wrapping_neg());
    let poly = (clipped.wrapping_add(qb)).wrapping_mul(clipped.wrapping_add(qb)).wrapping_add(qc);
    q.wrapping_mul(sign * poly + qc)
}

fn iexp(q: i32, qb: i32, qc: i32, qln2: i32, qln2_inv: i32) -> i32 {
    let z = q.wrapping_neg().wrapping_mul(qln2_inv) >> 16;
    let qp = q.wrapping_add(z.wrapping_mul(qln2));
    let exp = (qp.wrapping_add(qb)).wrapping_mul(qp.wrapping_add(qb)).wrapping_add(qc);
    exp.wrapping_shr(z as u32)
}

/// The command of a block of a row that is not the row's last.
fn non_terminating(cmd: u64) -> u64 {
    match cmd {
        MEAN => SUM,
        INV_STDDEV => VARIANCE,
        INV_SUM_EXP => SUM_EXP,
        cmd => cmd,
    }
}

fn act_of(bits: u64, acc_act: u8) -> u8 {
    match bits {
        0 if acc_act == SOFTMAX => SOFTMAX,
        0 => NONE,
        1 => RELU,
        2 => LAYERNORM,
        _ => IGELU,
    }
}

impl Gemmini {
    pub fn new() -> Self {
        Gemmini {
            spad: vec![[0; DIM]; BANK_NUM * BANK_ROWS],
            accumulator: vec![[0; DIM]; ACC_ROWS],
            pe_state: [[0; DIM]; DIM],
            output_sp_addr: GARBAGE_ADDR,
            preload_sp_addr: GARBAGE_ADDR,
            preload_cols: 0,
            preload_rows: 0,
            output_cols: 0,
            output_rows: 0,
            weight_stationary: false,
            sys_act: NONE,
            sys_shift: 0,
            a_transpose: false,
            b_transpose: false,
            c_stride: 1,
            a_stride: 1,
            acc_act: NONE,
            acc_shift: 1.0,
            load_strides: [DIM as u64; 3],
            load_block_strides: [DIM as u16; 3],
            load_shrunks: [false; 3],
            load_scales: [1.0; 3],
            pixels_per_rows: [1; 3],
            store_stride: DIM as u64,
            pool_stride: 0,
            pool_size: 0,
            pool_out_dim: 0,
            pool_porows: 0,
            pool_pocols: 0,
            pool_orows: 0,
            pool_ocols: 0,
            pool_lpad: 0,
            pool_upad: 0,
            igelu_qb: 0,
            igelu_qc: 0,
            qln2: 0,
            qln2_inv: 0,
            norm_stat_id: 0,
            norm_sum: [0; NORM_STAT_IDS],
            norm_running_max: [0; NORM_STAT_IDS],
            norm_max: [0; NORM_STAT_IDS],
            norm_count: [0; NORM_STAT_IDS],
            norm_mean: [0; NORM_STAT_IDS],
            norm_inv_stddev: [0.0; NORM_STAT_IDS],
            norm_inv_sum_exp: [0.0; NORM_STAT_IDS],
            norm_reset: [true; NORM_STAT_IDS],
            loop_ws: [0; 10],
            loop_conv: [0; 12],
            conv_d_row: 0,
            conv_c_row: 0,
            loop_issued: [0; 2],
            counter_val: [0; NUM_COUNTERS],
            counter_snapshot: [0; NUM_COUNTERS],
            counter_external: [false; NUM_COUNTERS],
            counted: 0,
            snapshot_enable: false,
            timing: Timing::new(),
            buffer: Vec::new(),
        }
    }

    /// Moves `rows` x `cols` from DRAM at `dram_addr` to the scratchpad or the
    /// accumulator at `sp_addr` (row addressed), with the load state `state_id`.
    pub(crate) fn mvin(&mut self, guest: &mut GuestAccess<'_>, dram_addr: u64, sp_addr: u64, state_id: usize) {
        let accumulator = (sp_addr >> 31) & 1 != 0;
        let accumulate = (sp_addr >> 30) & 1 != 0;
        let base_row = (sp_addr & 0x1FFF_FFFF) as usize;
        let cols = ((sp_addr >> ADDR_LEN) & 0xFFFF) as usize;
        let rows = ((sp_addr >> (ADDR_LEN + 16)) & 0xFFFF) as usize;
        let zeros = dram_addr == 0;
        let stride = self.load_strides[state_id];
        let block_stride = self.load_block_strides[state_id] as usize;
        let shrunk = self.load_shrunks[state_id];
        let scale = self.load_scales[state_id];
        let pixels = self.pixels_per_rows[state_id] as usize;
        let size = if accumulator && !shrunk { 4 } else { 1 };
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.resize(cols * size, 0);
        let mut cycles = 0;
        for row in 0..rows {
            if zeros {
                cycles += 1;
            } else {
                guest.read_buffer(dram_addr + row as u64 * stride, &mut buffer);
                cycles += ((cols * size) as u64).div_ceil(BEAT_BYTES).max(1);
            }
            for col in 0..cols {
                let spad_row = base_row + row + col / DIM * block_stride;
                let spad_col = col % DIM;
                let value = if zeros {
                    0
                } else if size == 4 {
                    i32::from_le_bytes(buffer[col * 4..col * 4 + 4].try_into().unwrap())
                } else {
                    i32::from(round_saturate_i8(f32::from(buffer[col] as i8) * scale))
                };
                for pixel in 0..pixels.min(spad_row + 1) {
                    if accumulator {
                        let cell = &mut self.accumulator[spad_row - pixel][spad_col + pixel * cols];
                        *cell = if accumulate { cell.wrapping_add(value) } else { value };
                    } else {
                        self.spad[spad_row - pixel][spad_col + pixel * cols] = value as i8;
                    }
                    if col % DIM == 0 {
                        self.timing.write(accumulator, spad_row - pixel);
                    }
                }
            }
        }
        self.buffer = buffer;
        self.timing
            .issue(LOAD, cycles, if zeros { 0 } else { LOAD_LATENCY });
    }

    /// Moves rows of the scratchpad or the accumulator at `sp_addr` to DRAM
    /// at `dram_addr`: scaled, activated and pooled when out of the accumulator.
    pub(crate) fn mvout(&mut self, guest: &mut GuestAccess<'_>, dram_addr: u64, sp_addr: u64) {
        let accumulator = (sp_addr >> 31) & 1 != 0;
        let full = (sp_addr >> 29) & 1 != 0;
        let norm_cmd = (sp_addr >> 26) & 7;
        let base_row = (sp_addr & 0x3FF_FFFF) as usize;
        let cols = ((sp_addr >> ADDR_LEN) & 0xFFFF) as usize;
        let rows = ((sp_addr >> (ADDR_LEN + 16)) & 0xFFFF) as usize;
        if self.pool_stride == 0 {
            let size = if accumulator && full { 4 } else { 1 };
            let mut buffer = std::mem::take(&mut self.buffer);
            buffer.resize(cols * size, 0);
            let mut cycles = 0;
            for i in 0..rows {
                let mut should_write = true;
                for j in (0..cols).step_by(DIM) {
                    let spad_row = base_row + j / DIM * DIM + i;
                    let len = (cols - j).min(DIM);
                    let cmd = if j + DIM >= cols { norm_cmd } else { non_terminating(norm_cmd) };
                    should_write = if accumulator {
                        let row = self.accumulator[spad_row];
                        self.norm(&row[..len], cmd)
                    } else {
                        self.norm(&[], cmd)
                    };
                    self.timing.read(accumulator, spad_row);
                }
                if !should_write {
                    continue;
                }
                for j in 0..cols {
                    let spad_row = base_row + j / DIM * DIM + i;
                    let spad_col = j % DIM;
                    if accumulator {
                        let value = self.pre_activation(self.accumulator[spad_row][spad_col]);
                        if full {
                            buffer[j * 4..j * 4 + 4].copy_from_slice(&value.to_le_bytes());
                        } else {
                            buffer[j] = activation(acc_scale(value, self.acc_shift), self.acc_act) as u8;
                        }
                    } else {
                        buffer[j] = self.spad[spad_row][spad_col] as u8;
                    }
                }
                guest.write_buffer(dram_addr + i as u64 * self.store_stride, &buffer);
                cycles += ((cols * size) as u64).div_ceil(BEAT_BYTES).max(1);
            }
            self.buffer = buffer;
            self.timing.issue(STORE, cycles, STORE_LATENCY);
        } else {
            let (stride, size) = (self.pool_stride as i64, self.pool_size as i64);
            let (orows, ocols) = (self.pool_orows as i64, self.pool_ocols as i64);
            let (porows, pocols) = (self.pool_porows as i64, self.pool_pocols as i64);
            for orow in 0..orows {
                for ocol in 0..ocols {
                    self.timing.read(accumulator, base_row + (orow * ocols + ocol) as usize);
                }
            }
            for porow in 0..porows {
                for pocol in 0..pocols {
                    for poch in 0..cols {
                        let mut value = i8::MIN;
                        for wrow in 0..size {
                            for wcol in 0..size {
                                let orow = porow * stride + wrow - self.pool_upad as i64;
                                let ocol = pocol * stride + wcol - self.pool_lpad as i64;
                                let row = (base_row as i64 + orow * ocols + ocol) as usize;
                                let elem = if orow < 0 || ocol < 0 || orow >= orows || ocol >= ocols {
                                    0
                                } else if accumulator {
                                    activation(acc_scale(self.accumulator[row][poch], self.acc_shift), self.acc_act)
                                } else {
                                    self.spad[row][poch]
                                };
                                value = value.max(elem);
                            }
                        }
                        let address = dram_addr
                            + (porow * self.pool_out_dim as i64 + pocol) as u64 * self.store_stride
                            + poch as u64;
                        guest.write_buffer(address, &[value as u8]);
                    }
                }
            }
            let windows = (porows * pocols * size * size) as u64;
            self.timing.issue(STORE, windows.max(1), STORE_LATENCY);
        }
    }

    pub(crate) fn preload(&mut self, bd_addr: u64, c_addr: u64) {
        self.preload_sp_addr = bd_addr as u32;
        self.output_sp_addr = c_addr as u32;
        self.preload_cols = (bd_addr >> ADDR_LEN) as u16;
        self.preload_rows = (bd_addr >> (ADDR_LEN + 16)) as u16;
        self.output_cols = (c_addr >> ADDR_LEN) as u16;
        self.output_rows = (c_addr >> (ADDR_LEN + 16)) as u16;
        self.timing.issue(EXECUTE, 0, 0);
    }

    pub(crate) fn compute(&mut self, a_addr: u64, bd_addr: u64, preload: bool) {
        let a = a_addr as u32;
        let bd = bd_addr as u32;
        let a_cols = (a_addr >> ADDR_LEN) as u16 as usize;
        let a_rows = (a_addr >> (ADDR_LEN + 16)) as u16 as usize;
        let bd_cols = (bd_addr >> ADDR_LEN) as u16 as usize;
        let bd_rows = (bd_addr >> (ADDR_LEN + 16)) as u16 as usize;
        let ws = self.weight_stationary;
        if preload {
            assert!(
                self.preload_sp_addr == GARBAGE_ADDR || (self.preload_sp_addr >> 30) & 3 == 0,
                "Gemmini preloads from the accumulator are not supported"
            );
            let transpose = ws && self.b_transpose;
            for i in 0..DIM {
                for j in 0..DIM {
                    let (r, c) = if transpose { (j, i) } else { (i, j) };
                    self.pe_state[i][j] = if i < self.preload_rows as usize
                        && j < self.preload_cols as usize
                        && self.preload_sp_addr != GARBAGE_ADDR
                    {
                        i32::from(self.spad[self.preload_sp_addr as usize + r][c])
                    } else {
                        0
                    };
                }
            }
            if self.preload_sp_addr != GARBAGE_ADDR {
                let rows = if transpose { self.preload_cols } else { self.preload_rows };
                for r in 0..(rows as usize).min(DIM) {
                    self.timing.read(false, self.preload_sp_addr as usize + r);
                }
            }
        }
        let mut results = [[0i32; DIM]; DIM];
        if bd != GARBAGE_ADDR {
            for i in 0..bd_rows.min(DIM) {
                for j in 0..bd_cols.min(DIM) {
                    results[i][j] = i32::from(self.spad[bd as usize + i][j]);
                }
                self.timing.read(false, bd as usize + i);
            }
        }
        let a_stride = self.a_stride as usize;
        if a != GARBAGE_ADDR {
            let rows = if self.a_transpose { a_cols } else { a_rows };
            for r in 0..rows.min(DIM) {
                self.timing.read(false, a as usize + a_stride * r);
            }
        }
        for i in 0..DIM {
            for j in 0..DIM {
                for k in 0..DIM {
                    let value_a = if a != GARBAGE_ADDR && i < a_rows && k < a_cols {
                        let r = a_stride * if self.a_transpose { k } else { i };
                        let c = if self.a_transpose { i } else { k };
                        i32::from(self.spad[a as usize + r][c])
                    } else {
                        0
                    };
                    if ws {
                        results[i][j] = results[i][j].wrapping_add(value_a.wrapping_mul(self.pe_state[k][j]));
                    } else {
                        let b = if bd != GARBAGE_ADDR && k < bd_rows && j < bd_cols {
                            let (r, c) = if self.b_transpose { (j, k) } else { (k, j) };
                            i32::from(self.spad[bd as usize + r][c])
                        } else {
                            0
                        };
                        self.pe_state[i][j] = self.pe_state[i][j].wrapping_add(value_a.wrapping_mul(b));
                    }
                }
            }
        }
        let output = self.output_sp_addr;
        if output != GARBAGE_ADDR {
            let acc = (output >> 31) & 1 != 0;
            let accumulate = (output >> 30) & 1 != 0;
            let base = (output & 0x1FFF_FFFF) as usize;
            let c_stride = self.c_stride as usize;
            for i in 0..self.output_rows as usize {
                let row = base + c_stride * i;
                for j in 0..self.output_cols as usize {
                    let value = if ws { results[i][j] } else { self.pe_state[i][j] };
                    if acc {
                        let cell = &mut self.accumulator[row][j];
                        *cell = if accumulate { cell.wrapping_add(value) } else { value };
                    } else {
                        let shift = if ws { 0 } else { self.sys_shift };
                        self.spad[row][j] = activation(sys_shift(value, shift), self.sys_act);
                    }
                }
                self.timing.write(acc, row);
            }
        }
        let rows = if self.a_transpose { a_cols } else { a_rows };
        let weights = if preload { self.preload_rows as usize } else { 0 };
        self.timing
            .issue(EXECUTE, rows.max(weights).max(1) as u64, MESH_LATENCY);
    }

    pub(crate) fn config(&mut self, rs1: u64, rs2: u64) {
        match rs1 & 3 {
            0 => {
                let ws = (rs1 >> 2) & 1 != 0;
                let act = act_of((rs1 >> 3) & 3, self.acc_act);
                let shift = rs2 as u32;
                let a_transpose = (rs1 >> 8) & 1 != 0;
                let b_transpose = (rs1 >> 9) & 1 != 0;
                assert!(shift < 32, "Gemmini sys_shift {shift} out of range");
                if (rs1 >> 7) & 1 == 0 {
                    self.weight_stationary = ws;
                    self.sys_act = act;
                    self.sys_shift = shift;
                    self.a_transpose = a_transpose;
                    self.b_transpose = b_transpose;
                }
                self.c_stride = (rs2 >> 48) as u16;
                self.a_stride = (rs1 >> 16) as u16;
                assert!(
                    !(!ws && !a_transpose && b_transpose) && !(ws && a_transpose && b_transpose),
                    "Gemmini dataflow does not support this transpose"
                );
                self.timing.issue(EXECUTE, 1, 0);
            }
            1 => {
                let id = ((rs1 >> 3) & 3) as usize;
                self.load_strides[id] = rs2;
                self.load_block_strides[id] = (rs1 >> 16) as u16;
                self.load_scales[id] = f32::from_bits((rs1 >> 32) as u32);
                self.load_shrunks[id] = (rs1 >> 2) & 1 != 0;
                self.pixels_per_rows[id] = ((rs1 >> 8) as u8).max(1);
                self.timing.issue(LOAD, 1, 0);
            }
            2 => {
                self.store_stride = rs2 & 0xFFFF_FFFF;
                self.acc_act = act_of((rs1 >> 2) & 3, self.acc_act);
                self.acc_shift = f32::from_bits((rs2 >> 32) as u32);
                self.pool_stride = ((rs1 >> 4) & 3) as u8;
                self.pool_size = ((rs1 >> 6) & 3) as u8;
                self.pool_upad = ((rs1 >> 8) & 3) as u8;
                self.pool_lpad = ((rs1 >> 10) & 3) as u8;
                self.pool_out_dim = (rs1 >> 24) as u8;
                self.pool_porows = (rs1 >> 32) as u8;
                self.pool_pocols = (rs1 >> 40) as u8;
                self.pool_orows = (rs1 >> 48) as u8;
                self.pool_ocols = (rs1 >> 56) as u8;
                self.timing.issue(STORE, 1, 0);
            }
            _ => {
                self.norm_stat_id = ((rs1 >> 8) & 0xFF) as usize;
                if (rs1 >> 17) & 1 == 0 {
                    self.igelu_qb = rs2 as i32;
                    self.igelu_qc = (rs2 >> 32) as i32;
                    self.acc_act |= (((rs1 >> 16) & 1) << 2) as u8;
                    assert!(self.acc_act <= SOFTMAX, "Gemmini activation {} out of range", self.acc_act);
                    if (rs1 >> 18) & 1 != 0 {
                        self.qln2_inv = (rs1 >> 32) as i32;
                    } else {
                        self.qln2 = (rs1 >> 32) as i32;
                    }
                }
                self.timing.issue(STORE, 1, 0);
            }
        }
    }

    fn pre_activation(&self, value: i32) -> i32 {
        let id = self.norm_stat_id;
        match self.acc_act {
            IGELU => igelu(value, self.igelu_qb, self.igelu_qc),
            LAYERNORM => (value.wrapping_sub(self.norm_mean[id]) as f32 * self.norm_inv_stddev[id]) as i32,
            SOFTMAX => {
                let exp = iexp(
                    value.wrapping_sub(self.norm_max[id]),
                    self.igelu_qb,
                    self.igelu_qc,
                    self.qln2,
                    self.qln2_inv,
                );
                i32::from(acc_scale(exp, self.norm_inv_sum_exp[id]))
            }
            _ => value,
        }
    }

    /// Accumulates the statistics of a block of a row; true when the row is written out.
    fn norm(&mut self, x: &[i32], cmd: u64) -> bool {
        let id = self.norm_stat_id;
        if self.norm_reset[id] {
            self.norm_sum[id] = 0;
            self.norm_count[id] = 0;
            self.norm_running_max[id] = i32::MIN;
        }
        self.norm_reset[id] = matches!(cmd, RESET | MEAN | INV_STDDEV);
        match cmd {
            SUM | MEAN => {
                for value in x {
                    self.norm_sum[id] = self.norm_sum[id].wrapping_add(*value);
                }
                self.norm_count[id] += x.len() as i32;
            }
            VARIANCE | INV_STDDEV => {
                for value in x {
                    let d = value.wrapping_sub(self.norm_mean[id]);
                    self.norm_sum[id] = self.norm_sum[id].wrapping_add(d.wrapping_mul(d));
                }
                self.norm_count[id] += x.len() as i32;
            }
            MAX => {
                for value in x {
                    self.norm_running_max[id] = self.norm_running_max[id].max(*value);
                }
            }
            SUM_EXP | INV_SUM_EXP => {
                self.norm_max[id] = self.norm_running_max[id];
                for value in x {
                    let exp = iexp(
                        value.wrapping_sub(self.norm_max[id]),
                        self.igelu_qb,
                        self.igelu_qc,
                        self.qln2,
                        self.qln2_inv,
                    );
                    self.norm_sum[id] = self.norm_sum[id].wrapping_add(exp);
                }
            }
            _ => {}
        }
        match cmd {
            MEAN => self.norm_mean[id] = self.norm_sum[id] / self.norm_count[id],
            INV_STDDEV => {
                let variance = self.norm_sum[id] / self.norm_count[id];
                let stddev = if variance == 0 { 1 } else { f64::from(variance).sqrt() as i32 };
                self.norm_inv_stddev[id] = 1.0 / stddev as f32;
            }
            INV_SUM_EXP => {
                self.norm_running_max[id] = i32::MIN;
                self.norm_inv_sum_exp[id] = 127.0 / self.norm_sum[id] as f32;
            }
            _ => {}
        }
        cmd == RESET
    }

    fn counter(&mut self, rs1: u64, cycle: u64) -> u64 {
        let busy = self.timing.busy(cycle);
        for i in 0..NUM_COUNTERS {
            self.counter_val[i] = self.counter_val[i].wrapping_add((busy - self.counted) as u32);
        }
        self.counted = busy;
        let index = ((rs1 >> 4) & 7) as usize;
        let pending = self.timing.pending(LOAD, cycle) as u32;
        let value = |gemmini: &Self, i: usize| {
            if gemmini.counter_external[i] {
                pending
            } else {
                gemmini.counter_val[i]
            }
        };
        if rs1 & 1 != 0 {
            self.counter_val = [0; NUM_COUNTERS];
        }
        if rs1 & 2 != 0 {
            self.snapshot_enable = false;
        }
        if rs1 & 4 != 0 {
            self.snapshot_enable = true;
            self.counter_snapshot = std::array::from_fn(|i| value(self, i));
        }
        if rs1 & 8 != 0 {
            self.counter_val[index] = 0;
            self.counter_external[index] = (rs1 >> 31) & 1 != 0;
        }
        u64::from(if self.snapshot_enable {
            self.counter_snapshot[index]
        } else {
            value(self, index)
        })
    }

    /// A loop instruction enters the unroller when one of its two slots is
    /// free; its instructions follow the instructions presented before it.
    fn unroll(&mut self, guest: &mut GuestAccess<'_>, funct: u32, rs1: u64, rs2: u64, cycle: u64) {
        let accepted = cycle.max(self.loop_issued[0]);
        self.timing.cursor = self.timing.cursor.max(accepted);
        if funct == 8 {
            self.run_loop_ws(guest, rs1, rs2);
        } else {
            self.run_loop_conv_ws(guest, rs1, rs2);
        }
        self.loop_issued = [self.loop_issued[1], self.timing.cursor];
        self.timing.accepted = accepted;
    }

    /// The unroller presents its next instruction a cycle after the last.
    pub(crate) fn next(&mut self) {
        self.timing.cursor += 1;
    }
}

impl Rocc for Gemmini {
    fn cycles(&self) -> (u64, u64) {
        (self.timing.accepted, self.timing.complete)
    }

    fn custom(&mut self, mut guest: GuestAccess<'_>, request: CustomInstruction, cycle: u64) -> Result<u64, Trap> {
        let (rs1, rs2) = (request.rs1, request.rs2);
        let funct = request.instruction >> 25;
        self.timing.cursor = self.timing.cursor.max(cycle);
        self.timing.accepted = cycle;
        match funct {
            0 => self.config(rs1, rs2),
            1 => self.mvin(&mut guest, rs1, rs2, 1),
            2 => self.mvin(&mut guest, rs1, rs2, 0),
            14 => self.mvin(&mut guest, rs1, rs2, 2),
            3 => self.mvout(&mut guest, rs1, rs2),
            4 => self.compute(rs1, rs2, true),
            5 => self.compute(rs1, rs2, false),
            6 => self.preload(rs1, rs2),
            7 | 127 => {}
            8 | 15 => self.unroll(&mut guest, funct, rs1, rs2, cycle),
            9..=13 => {
                self.timing.accepted = cycle.max(self.loop_issued[0]);
                self.loop_ws[(funct as usize - 9) * 2] = rs1;
                self.loop_ws[(funct as usize - 9) * 2 + 1] = rs2;
            }
            16..=21 => {
                self.timing.accepted = cycle.max(self.loop_issued[0]);
                self.loop_conv[(funct as usize - 16) * 2] = rs1;
                self.loop_conv[(funct as usize - 16) * 2 + 1] = rs2;
            }
            126 => return Ok(self.counter(rs1, cycle)),
            _ => return Err(Trap::illegal(request.instruction)),
        }
        Ok(0)
    }

    fn barrier(&self) -> bool {
        false
    }
}
