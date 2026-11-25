// The cycles of one MAXPOOL or TOPK_U32 instruction from its command handshake
// to the first cycle its response is valid, against an ideal outside world:
// every bank request ready, a read answered the next cycle, a write
// acknowledged the next cycle. Delays outside the Ball belong to the bank and
// bus model, not here. From MaxPoolBall.scala. The UVM tests of the RTL check
// both exactly.

// MAXPOOL, outSide^2 outputs in each of `groups` groups:
//   1           idle -> scan: the first scan at cycle 1
//   2 / valid   scan raises the read of an in-bounds window position, readResp
//               takes it the next cycle (-> scan, or -> writeReq on the last)
//   1 / padding scan steps the kernel over a padding position (or, on the
//               last position, -> writeReq) without a read
//   2 / output  writeReq, writeResp -> scan (a new group starts like a new
//               output)
//   complete    the response is valid the cycle after the last writeResp.

/// One MAXPOOL as the Ball receives it: rs2 packs inSide, outSide, kernel,
/// stride, padding (4 bits each from bit 0), startRow [41:38] and startCol
/// [45:42]; `groups` is op1_col.
pub fn maxpool_cycles(rs2: u64, groups: u64) -> u64 {
    let input_side = rs2 & 0xf;
    let output_side = (rs2 >> 4) & 0xf;
    let kernel = (rs2 >> 8) & 0xf;
    let stride = (rs2 >> 12) & 0xf;
    let padding = (rs2 >> 16) & 0xf;
    let start_row = (rs2 >> 38) & 0xf;
    let start_col = (rs2 >> 42) & 0xf;
    let mut group = 0;
    for output_y in 0..output_side {
        for output_x in 0..output_side {
            group += 2;
            for kernel_y in 0..kernel {
                for kernel_x in 0..kernel {
                    let y = output_y * stride + kernel_y + start_row;
                    let x = output_x * stride + kernel_x + start_col;
                    let valid = y >= padding && x >= padding && y < padding + input_side && x < padding + input_side;
                    group += if valid { 2 } else { 1 };
                }
            }
        }
    }
    groups * group + 1
}

// TOPK_U32, `iter` rows of 4 keys in each of `groups` groups, k results; a
// pass finds the next 4 of the remaining keys:
//   1            idle -> topScan
//   iter + 2     per group of a pass, topScan: one read request per cycle from
//                the group's first cycle, its response the next cycle
//                (topRowReady), sorted into topBest one cycle later
//                (topRowValid); the group's last topRowValid restarts the
//                requests in the next group the cycle after, or -> write
//   2 / write    topWriteReq, topWriteResp: two rows (4 pairs) per pass while
//                topRank + 2 < k, else one, then -> topScan or complete
//   complete     the response is valid the cycle after the last write.

/// One TOPK_U32 as the Ball receives it: `iter` rows, `groups` (op1_col), k
/// (rs2[47:32]).
pub fn topk_cycles(iter: u64, groups: u64, k: u64) -> u64 {
    let mut cycles = 1;
    for pass in 0..k.div_ceil(4) {
        let writes = if 4 * pass + 2 < k { 2 } else { 1 };
        cycles += groups * (iter + 2) + 2 * writes;
    }
    cycles
}
