use crate::gemmini::{
    Gemmini, ACC_ROWS, ADDR_LEN, BANK_NUM, BANK_ROWS, DIM, GARBAGE_ADDR, LAYERNORM, MAX_BLOCK_LEN, MAX_BLOCK_LEN_ACC,
    SOFTMAX,
};
use bebop_bemu::GuestAccess;

const D: u64 = DIM as u64;

/// The bits of 1.0f32, the identity scale of mvin.
const SCALE_IDENTITY: u64 = 0x3f80_0000;

fn rows_cols(rows: u64, cols: u64) -> u64 {
    (rows << 48) | (cols << 32)
}

/// The loop unrollers (LoopMatmul and LoopConv) issue the instructions of a
/// loop as the Spike reference unrolls them.
impl Gemmini {
    pub(crate) fn run_loop_ws(&mut self, guest: &mut GuestAccess<'_>, rs1: u64, rs2: u64) {
        let ex_accumulate = rs1 & 1 != 0;
        let full_c = (rs1 >> 1) & 1;
        let low_d = (rs1 >> 2) & 1 != 0;
        let act = ((rs1 >> 8) & 7) as u8;
        let a_transpose = rs2 & 1 != 0;
        let b_transpose = (rs2 >> 1) & 1 != 0;
        let a_spad_id = (rs1 >> 18) & 3;
        let b_spad_id = (rs1 >> 16) & 3;
        let is_resadd = (rs2 >> 2) & 1 != 0;
        let [pads, bounds, a, b, d, c, a_stride, b_stride, d_stride, c_stride] = self.loop_ws;
        let (i_bound, j_bound, k_bound) = (bounds & 0xFFFF, (bounds >> 16) & 0xFFFF, (bounds >> 32) & 0xFFFF);
        let (pad_i, pad_j, pad_k) = (pads & 0xFFFF, (pads >> 16) & 0xFFFF, (pads >> 32) & 0xFFFF);
        let total_spad_rows = (i_bound * k_bound + k_bound * j_bound) * D;
        let total_acc_rows = i_bound * j_bound * D;
        assert!(
            total_spad_rows <= (BANK_NUM * BANK_ROWS / 2) as u64 && total_acc_rows <= (ACC_ROWS / 2) as u64,
            "LOOP_WS bounds were too large for double-buffering"
        );
        let half = (BANK_NUM * BANK_ROWS / 2) as u64;
        let mut a_start = 0u64;
        let mut b_start = half - k_bound * j_bound * D;
        let d_start = 1u64 << (ADDR_LEN - 1);
        let c_start = (3u64 << (ADDR_LEN - 2)) | (full_c << (ADDR_LEN - 3));
        if a_spad_id == 2 {
            a_start = half;
        }
        if b_spad_id == 2 {
            b_start = 2 * half - k_bound * j_bound * D;
        }
        let pad = |index: u64, bound: u64, pad: u64| D - if index == bound - 1 { pad } else { 0 };
        let sizeof_c = if full_c != 0 { 4 } else { 1 };
        if is_resadd {
            a_start = 1 << (ADDR_LEN - 1);
            b_start = 3 << (ADDR_LEN - 2);
            for i in 0..i_bound {
                for j in 0..j_bound {
                    let offset = (i * j_bound + j) * D;
                    let cols = pad(j, j_bound, pad_j);
                    let rows = pad(i, i_bound, pad_i);
                    self.next();
                    self.mvin(guest, a + (i * a_stride + j) * D, rows_cols(rows, cols) | (a_start + offset), 0);
                    self.next();
                    self.mvin(guest, b + (i * b_stride + j) * D, rows_cols(rows, cols) | (b_start + offset), 1);
                    if c != 0 {
                        self.next();
                        self.mvout(
                            guest,
                            c + (i * c_stride + j) * D * sizeof_c,
                            rows_cols(rows, cols) | (c_start + offset),
                        );
                    }
                }
            }
            return;
        }
        if d != 0 {
            let sizeof_d = if low_d { 1 } else { 4 };
            for i in 0..i_bound {
                for j in 0..j_bound {
                    let rows = pad(i, i_bound, pad_i);
                    let cols = pad(j, j_bound, pad_j);
                    self.next();
                    self.mvin(
                        guest,
                        d + (i * d_stride + j) * D * sizeof_d,
                        rows_cols(rows, cols) | (d_start + (i * j_bound + j) * D),
                        2,
                    );
                }
            }
        }
        // LoopMatmul's load FSMs run ahead of its execute and store FSMs: the
        // loads of the loop are issued first.
        for k in 0..k_bound {
            for j in 0..j_bound {
                for i in 0..i_bound {
                    let a_sp = if a_transpose {
                        a_start + (k * i_bound + i) * D
                    } else {
                        a_start + (i * k_bound + k) * D
                    };
                    let b_sp = if b_transpose {
                        b_start + (j * k_bound + k) * D
                    } else {
                        b_start + (k * j_bound + j) * D
                    };
                    if j == 0 && a != 0 {
                        let (dram, cols, rows) = if a_transpose {
                            (a + (k * a_stride + i) * D, pad(i, i_bound, pad_i), pad(k, k_bound, pad_k))
                        } else {
                            (a + (i * a_stride + k) * D, pad(k, k_bound, pad_k), pad(i, i_bound, pad_i))
                        };
                        self.next();
                        self.mvin(guest, dram, rows_cols(rows, cols) | a_sp, 0);
                    }
                    if i == 0 && b != 0 {
                        let (dram, cols, rows) = if b_transpose {
                            (b + (j * b_stride + k) * D, pad(k, k_bound, pad_k), pad(j, j_bound, pad_j))
                        } else {
                            (b + (k * b_stride + j) * D, pad(j, j_bound, pad_j), pad(k, k_bound, pad_k))
                        };
                        self.next();
                        self.mvin(guest, dram, rows_cols(rows, cols) | b_sp, 1);
                    }
                }
            }
        }
        for k in 0..k_bound {
            for j in 0..j_bound {
                for i in 0..i_bound {
                    let a_sp = if a_transpose {
                        a_start + (k * i_bound + i) * D
                    } else {
                        a_start + (i * k_bound + k) * D
                    };
                    let b_sp = if b_transpose {
                        b_start + (j * k_bound + k) * D
                    } else {
                        b_start + (k * j_bound + j) * D
                    };
                    let c_sp = c_start + (i * j_bound + j) * D;
                    let pre_sp = if i == 0 { b_sp } else { u64::from(GARBAGE_ADDR) };
                    let mut out_sp = c_sp;
                    if !ex_accumulate && k == 0 {
                        out_sp &= !(1 << (ADDR_LEN - 2));
                    }
                    let a_cols = pad(k, k_bound, pad_k);
                    let a_rows = pad(i, i_bound, pad_i);
                    let b_cols = pad(j, j_bound, pad_j);
                    let b_rows = pad(k, k_bound, pad_k);
                    let c_cols = pad(j, j_bound, pad_j);
                    let c_rows = pad(i, i_bound, pad_i);
                    self.next();
                    self.preload(rows_cols(b_rows, b_cols) | pre_sp, rows_cols(c_rows, c_cols) | out_sp);
                    self.next();
                    self.compute(
                        rows_cols(a_rows, a_cols) | a_sp,
                        rows_cols(D, D) | u64::from(GARBAGE_ADDR),
                        i == 0,
                    );
                    if c != 0 && k == k_bound - 1 {
                        if act != LAYERNORM && act != SOFTMAX {
                            self.next();
                            self.mvout(
                                guest,
                                c + (i * c_stride + j) * D * sizeof_c,
                                rows_cols(c_rows, c_cols) | c_sp,
                            );
                        } else if j == j_bound - 1 {
                            let cmds: [[u64; 2]; 3] = if act == LAYERNORM {
                                [[1, 2], [3, 4], [0, 0]]
                            } else {
                                [[5, 5], [6, 7], [0, 0]]
                            };
                            let rows = pad(i, i_bound, pad_i);
                            for row in (0..rows).step_by(4) {
                                let stat_ids = (rows - row).min(4);
                                for cmd in cmds {
                                    for stat_id in 0..stat_ids {
                                        self.next();
                                        self.config((1 << 17) | (stat_id << 8) | 3, 0);
                                        let r = row + stat_id;
                                        for jj in 0..j_bound {
                                            let last = jj + 1 == j_bound;
                                            let norm_sp = c_start
                                                + (i * j_bound + jj) * D
                                                + r
                                                + ((if last { cmd[1] } else { cmd[0] }) << 26);
                                            let dram = c
                                                + (i * c_stride + jj) * D * sizeof_c
                                                + r * c_stride * sizeof_c;
                                            let cols = D - if last { pad_j } else { 0 };
                                            self.next();
                                            self.mvout(guest, dram, rows_cols(1, cols) | norm_sp);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn run_loop_conv_ws(&mut self, guest: &mut GuestAccess<'_>, rs1: u64, rs2: u64) {
        let no_bias = rs1 & 1 != 0;
        let wrot180 = (rs1 >> 1) & 1 != 0;
        let trans_output_1203 = (rs1 >> 2) & 1 != 0;
        let trans_weight_1203 = (rs1 >> 3) & 1 != 0;
        let trans_weight_0132 = (rs1 >> 4) & 1 != 0;
        let trans_input_3120 = (rs1 >> 5) & 1 != 0;
        let dw = (rs1 >> 6) & 1 != 0;
        let max_pixels_per_row = ((rs1 >> 8) & 0xFF).max(1) as i64;
        let no_pool = rs2 & 1 != 0;
        let downsample = ((rs2 >> 1) & 1) as i64;
        let input_dilated = ((rs2 >> 2) & 1) as i64;
        // The reference keeps the activation as a bool: any activation is ReLU.
        let activation = u64::from((rs2 >> 3) & 3 != 0);
        let a_spad_id = (rs1 >> 18) & 3;
        let b_spad_id = (rs1 >> 16) & 3;

        let [c1, c2, c3, c4, c5, c6, c7, c8, weights, output, bias, input] = self.loop_conv;
        let f = |value: u64, shift: u32, mask: u64| ((value >> shift) & mask) as i64;
        let batch_size = f(c1, 0, 0xFFFF);
        let in_row_dim = f(c1, 16, 0xFFFF);
        let in_channels = f(c1, 32, 0xFFFF);
        let out_channels = f(c1, 48, 0xFFFF);
        let out_row_dim = f(c2, 0, 0xFFFF);
        let pool_out_row_dim = f(c2, 16, 0xFFFF);
        let out_col_dim = f(c2, 32, 0xFFFF);
        let stride = f(c2, 48, 0xFF);
        let kernel_dim = f(c3, 48, 0xFFFF);
        let pool_out_col_dim = f(c3, 32, 0xFFFF);
        let pool_size = f(c3, 16, 0xFFFF);
        let pool_stride = f(c3, 8, 0xFF);
        let batches = f(c4, 48, 0xFFFF);
        let porows = f(c4, 32, 0xFFFF);
        let pocols = f(c4, 16, 0xFFFF);
        let pochs = f(c4, 0, 0xFFFF);
        let krows = f(c5, 48, 0xFFFF);
        let kcols = f(c5, 32, 0xFFFF);
        let kchs = f(c5, 16, 0xFFFF);
        let lpad = f(c5, 0, 0xFFFF);
        let rpad = f(c6, 48, 0xFFFF);
        let upad = f(c6, 32, 0xFFFF);
        let dpad = f(c6, 24, 0xFF);
        let plpad = f(c6, 16, 0xFF);
        let in_col_dim = f(c6, 0, 0xFFFF);
        let orows = f(c7, 48, 0xFFFF);
        let pupad = f(c7, 21, 0xFF);
        let kernel_dilation = f(c7, 0, 0xFF);
        let in_stride = f(c8, 48, 0xFFFF);
        let weight_stride = f(c8, 32, 0xFFFF);
        let out_stride = f(c8, 16, 0xFFFF);
        let ocols = f(c8, 0, 0xFFFF);
        let ochs = pochs;
        let dim = DIM as i64;
        assert!(input_dilated == 0 || stride == 1, "dilated LOOP_CONV_WS input needs stride 1");

        let dilated_krows = krows + (kernel_dilation - 1) * (krows - 1);
        let dilated_kcols = kcols + (kernel_dilation - 1) * (kcols - 1);
        let irows_without_dilation = orows * stride + dilated_krows - 1;
        let icols_without_dilation = ocols * stride + dilated_kcols - 1;
        let irows_unpadded_without_dilation = irows_without_dilation - upad - dpad;
        let icols_unpadded_without_dilation = icols_without_dilation - lpad - rpad;
        let ichs = kchs;
        let undilated = |x: i64| (x + input_dilated) >> input_dilated;
        let irows_unpadded = if input_dilated != 0 {
            (irows_unpadded_without_dilation + 1) / 2
        } else {
            irows_unpadded_without_dilation
        };
        let icols_unpadded = if input_dilated != 0 {
            (icols_unpadded_without_dilation + 1) / 2
        } else {
            icols_unpadded_without_dilation
        };
        let irows = if input_dilated != 0 {
            irows_unpadded + undilated(upad) + undilated(dpad)
        } else {
            irows_without_dilation
        };
        let icols = if input_dilated != 0 {
            icols_unpadded + undilated(lpad) + undilated(rpad)
        } else {
            icols_without_dilation
        };
        let out_channels_per_bank = ochs / dim + i64::from(ochs % dim != 0);
        let in_channels_per_bank = kchs / dim + i64::from(kchs % dim != 0);
        let b_rows = if trans_weight_0132 {
            in_channels_per_bank * kcols * krows * ochs
        } else {
            out_channels_per_bank * kcols * krows * kchs
        };
        let half = (BANK_NUM * BANK_ROWS / 2) as i64;
        let mut a_start = 0i64;
        let mut b_start = half - b_rows;
        let d_start = (1i64 << (ADDR_LEN - 1)) + i64::from(self.conv_d_row);
        let c_start = (3i64 << (ADDR_LEN - 2)) + i64::from(self.conv_c_row);
        if a_spad_id == 2 {
            a_start = half;
        }
        if b_spad_id == 2 {
            b_start = 2 * half - b_rows;
        }
        if bias != 0 {
            self.conv_d_row = (self.conv_d_row + ACC_ROWS as u32 / 2) % ACC_ROWS as u32;
        }
        if output != 0 {
            self.conv_c_row = (self.conv_c_row + ACC_ROWS as u32 / 2) % ACC_ROWS as u32;
        }
        let addr = |rows: i64, cols: i64, sp: i64| ((rows as u64) << 48) | ((cols as u64) << 32) | sp as u32 as u64;

        if bias != 0 {
            let max_ochs_per_mvin = ochs.min((MAX_BLOCK_LEN_ACC * DIM) as i64);
            self.next();
            self.config(
                (SCALE_IDENTITY << 32) | (((batches * orows * ocols) as u64) << 16) | (1 << 8) | (2 << 3) | 1,
                0,
            );
            for b in 0..batches {
                for orow in 0..orows {
                    for ocol in (0..ocols).step_by(DIM) {
                        let i = (ocols - ocol).min(dim);
                        for och in (0..ochs).step_by(max_ochs_per_mvin as usize) {
                            let j = (ochs - och).min(max_ochs_per_mvin);
                            let d_sp = d_start
                                + (och / dim) * batches * orows * ocols
                                + b * orows * ocols
                                + orow * ocols
                                + ocol;
                            self.next();
                            self.mvin(
                                guest,
                                if no_bias { 0 } else { bias + och as u64 * 4 },
                                addr(i, j, d_sp),
                                2,
                            );
                        }
                    }
                }
            }
        }

        let ds = |x: i64| x >> downsample;
        let us = |x: i64| x << downsample;

        if input != 0 {
            let max_chs_per_mvin = if trans_input_3120 {
                batches.min((MAX_BLOCK_LEN * DIM) as i64)
            } else {
                ichs.min((MAX_BLOCK_LEN * DIM) as i64)
            };
            let dram_stride = if trans_input_3120 { batch_size } else { in_stride };
            let spad_stride = if trans_input_3120 {
                ichs * ds(irows) * ds(icols)
            } else {
                batches * ds(irows) * ds(icols)
            };
            self.next();
            self.config(
                (SCALE_IDENTITY << 32) | ((spad_stride as u64 & 0xFFFF) << 16) | ((max_pixels_per_row as u64) << 8) | 1,
                us(dram_stride) as u64,
            );
            let b_it = if trans_input_3120 { max_chs_per_mvin } else { 1 };
            let ich_it = if trans_input_3120 { 1 } else { max_chs_per_mvin };
            let mut b = 0;
            while b < batches {
                let mut irow = -undilated(upad);
                while irow < irows_unpadded + undilated(dpad) {
                    let irow_padded = irow + undilated(upad);
                    let mut icol = -undilated(lpad);
                    while icol < icols_unpadded + undilated(rpad) {
                        let mut i = (icols_unpadded - icol).min(us(dim));
                        if icol < 0 {
                            i = (-icol).min(dim);
                        } else if icol >= icols_unpadded {
                            i = (icols_unpadded + undilated(rpad) - icol).min(dim);
                        }
                        let icol_padded = icol + undilated(lpad);
                        let mut ich = 0;
                        while ich < ichs {
                            let k = if trans_input_3120 {
                                (batches - b).min(max_chs_per_mvin)
                            } else {
                                (ichs - ich).min(max_chs_per_mvin)
                            };
                            let a_sp = if trans_input_3120 {
                                a_start
                                    + (b / dim) * spad_stride
                                    + ich * ds(irows) * ds(icols)
                                    + ds(irow_padded) * ds(icols)
                                    + ds(icol_padded)
                            } else {
                                a_start
                                    + (ich / dim) * spad_stride
                                    + b * ds(irows) * ds(icols)
                                    + ds(irow_padded) * ds(icols)
                                    + ds(icol_padded)
                            };
                            let zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;
                            let dram = if zeros {
                                0
                            } else if trans_input_3120 {
                                input
                                    .wrapping_add((((ich * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b)) as u64)
                            } else {
                                input.wrapping_add(
                                    ((b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_stride + ich) as u64,
                                )
                            };
                            self.next();
                            self.mvin(guest, dram, addr(ds(i), k, a_sp), 0);
                            ich += ich_it;
                        }
                        icol += i;
                    }
                    irow += us(1);
                }
                b += b_it;
            }
        }

        if weights != 0 {
            let max_chs_per_mvin = if trans_weight_0132 {
                kchs.min((MAX_BLOCK_LEN * DIM) as i64)
            } else {
                ochs.min((MAX_BLOCK_LEN * DIM) as i64)
            };
            let dram_stride = if dw {
                1
            } else if trans_weight_1203 {
                kernel_dim * kernel_dim * out_channels
            } else if trans_weight_0132 {
                in_channels
            } else {
                weight_stride
            };
            let spad_block_stride = if trans_weight_0132 {
                krows * kcols * ochs
            } else {
                krows * kcols * kchs
            };
            self.next();
            self.config(
                (SCALE_IDENTITY << 32) | ((spad_block_stride as u64 & 0xFFFF) << 16) | (1 << 8) | (1 << 3) | 1,
                dram_stride as u64,
            );
            let och_it = if trans_weight_0132 { dim } else { max_chs_per_mvin };
            let kch_it = if trans_weight_0132 { max_chs_per_mvin } else { dim };
            let mut och = 0;
            while och < ochs {
                for krow in 0..krows {
                    for kcol in 0..kcols {
                        let mut kch = 0;
                        while kch < kchs {
                            let (k, j) = if trans_weight_0132 {
                                ((ochs - och).min(dim), (kchs - kch).min(max_chs_per_mvin))
                            } else {
                                ((kchs - kch).min(dim), (ochs - och).min(max_chs_per_mvin))
                            };
                            let b_sp = if trans_weight_0132 {
                                b_start + (kch / dim) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och
                            } else {
                                b_start + (och / dim) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch
                            };
                            let w = if dw {
                                weights + (krow * kernel_dim + kcol) as u64
                            } else if trans_weight_1203 {
                                weights
                                    + ((kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och)
                                        as u64
                            } else if trans_weight_0132 {
                                weights
                                    + ((krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch)
                                        as u64
                            } else {
                                weights
                                    + ((krow * kernel_dim * in_channels + kcol * in_channels + kch) * weight_stride + och)
                                        as u64
                            };
                            self.next();
                            self.mvin(guest, w, addr(k, j, b_sp), 1);
                            kch += kch_it;
                        }
                    }
                }
                och += och_it;
            }
        }

        {
            let b_it = if trans_input_3120 { dim } else { 1 };
            let ocol_it = if trans_input_3120 { 1 } else { dim << input_dilated };
            if trans_input_3120 {
                let a_stride = (irows * icols) as u64 & 0xFFFF;
                let c_stride = (orows * ocols) as u64 & 0xFFFF;
                self.next();
                self.config((a_stride << 16) | (1 << 7), c_stride << 48);
            }
            for och in (0..ochs).step_by(DIM) {
                for krow in 0..krows {
                    for kcol in (0..kcols).step_by(max_pixels_per_row as usize) {
                        for kch in (0..kchs).step_by(DIM) {
                            let mut new_weights = true;
                            let mut b = 0;
                            while b < batches {
                                for orow in 0..orows {
                                    if input_dilated != 0 && (krow * kernel_dilation + orow - upad) % 2 != 0 {
                                        continue;
                                    }
                                    let mut ocol = 0;
                                    while ocol < ocols {
                                        if input_dilated != 0 && (kcol * kernel_dilation + ocol - lpad) % 2 != 0 {
                                            ocol += 1;
                                            continue;
                                        }
                                        let irow = undilated(orow * stride + krow * kernel_dilation);
                                        let icol = undilated(ocol * stride + kcol * kernel_dilation);
                                        let c_sp = c_start
                                            + (och / dim) * batches * orows * ocols
                                            + b * orows * ocols
                                            + orow * ocols
                                            + ocol;
                                        let pixels = (kcols - kcol).min(max_pixels_per_row);
                                        let mut i = undilated((ocols - ocol).min(dim << input_dilated));
                                        let j = (ochs - och).min(dim);
                                        let k = pixels * (kchs - kch).min(dim);
                                        if trans_input_3120 {
                                            i = (batches - b).min(dim);
                                        }
                                        let a_sp = if trans_input_3120 {
                                            a_start
                                                + (b / dim) * kchs * ds(irows) * ds(icols)
                                                + kch * ds(irows) * ds(icols)
                                                + ds(irow) * ds(icols)
                                                + ds(icol)
                                        } else {
                                            a_start
                                                + (kch / dim) * batches * ds(irows) * ds(icols)
                                                + b * ds(irows) * ds(icols)
                                                + ds(irow) * ds(icols)
                                                + ds(icol)
                                        };
                                        let krow_ = if wrot180 { krows - krow - 1 } else { krow };
                                        let kcol_ = if wrot180 { kcols - kcol - 1 } else { kcol };
                                        let b_sp = if trans_weight_0132 {
                                            b_start
                                                + (kch / dim) * krows * kcols * ochs
                                                + krow_ * kcols * ochs
                                                + kcol_ * ochs
                                                + och
                                        } else {
                                            b_start
                                                + (och / dim) * krows * kcols * kchs
                                                + krow_ * kcols * kchs
                                                + kcol_ * kchs
                                                + kch
                                        };
                                        let pre_sp = if new_weights { b_sp } else { i64::from(GARBAGE_ADDR) };
                                        self.next();
                                        self.preload(addr(k, j, pre_sp), addr(i, j, c_sp));
                                        self.next();
                                        self.compute(addr(i, k, a_sp), addr(i, j, i64::from(GARBAGE_ADDR)), new_weights);
                                        ocol += ocol_it;
                                        new_weights = false;
                                    }
                                }
                                b += b_it;
                            }
                        }
                    }
                }
            }
        }

        if output != 0 && no_pool {
            for b in 0..batches {
                for orow in 0..orows {
                    for ocol in (0..ocols).step_by(DIM) {
                        let i = (ocols - ocol).min(dim);
                        for och in (0..ochs).step_by(DIM) {
                            let j = (ochs - och).min(dim);
                            let c_sp =
                                c_start + (och / dim) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;
                            let out = if trans_output_1203 {
                                output
                                    + ((orow * out_col_dim * batch_size + ocol * batch_size + b) * out_channels + och)
                                        as u64
                            } else {
                                output
                                    + ((b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * out_stride + och)
                                        as u64
                            };
                            self.next();
                            self.mvout(guest, out, addr(i, j, c_sp));
                        }
                    }
                }
            }
        } else if output != 0 {
            let scale = u64::from(self.acc_shift.to_bits());
            self.next();
            self.config(
                ((ocols as u64) << 56)
                    | ((orows as u64) << 48)
                    | ((pocols as u64) << 40)
                    | ((porows as u64) << 32)
                    | ((pool_out_col_dim as u64) << 24)
                    | ((plpad as u64) << 10)
                    | ((pupad as u64) << 8)
                    | ((pool_size as u64) << 6)
                    | ((pool_stride as u64) << 4)
                    | (activation << 2)
                    | 2,
                (scale << 32) | out_stride as u64,
            );
            for b in 0..batches {
                for poch in (0..pochs).step_by(DIM) {
                    let channels = if poch + dim >= pochs { pochs - poch } else { dim };
                    let c_sp = c_start + (poch / dim) * batches * orows * ocols + b * orows * ocols;
                    self.next();
                    self.mvout(
                        guest,
                        output + ((b * pool_out_row_dim * pool_out_col_dim) * out_stride + poch) as u64,
                        addr(0, channels, c_sp),
                    );
                }
            }
            self.next();
            self.config((activation << 2) | 2, (scale << 32) | out_stride as u64);
        }
    }
}
