// The cycles of one IM2COL instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle and the port taking no new write until that acknowledgement is
// accepted. Delays outside the Ball belong to the bank and bus model, not
// here. From Im2col.scala and RestoringDiv.scala:
//   70     four 16-bit RestoringDiv runs, each started in the cycle the last
//          finishes: start, 16 busy cycles, done, so done at 17, 34, 51, 68
//          (divRows, divCols, divStart, divTiles); check at 69; the first
//          select at 70
//   rows   N = kernel^2 * input groups * paddedWindows. A row that is read
//          takes select (read request) and readResp (enqueue into the 2-entry
//          pendingRows) the next cycle; a padding row or a window past
//          windowCount is enqueued by select itself in one cycle. The writer
//          takes the queue front the cycle after an enqueue: write request,
//          acknowledgement (dequeue) the next cycle, the port free the cycle
//          after: one row per 2 cycles. Reads feed it exactly at that rate,
//          padding rows faster, so from the first write (cycle W) the writer
//          never waits: the last acknowledgement at W + 2 * N - 1, the
//          response valid at W + 2 * N.
//   W      the cycle after the first enqueue: 72 if the first row (K tile 0,
//          window windowStart) is read, 71 if it is padding.
// The UVM tests of the RTL check it exactly.

/// One IM2COL as the Ball receives it: an `iter` x `iter` input, rs2 packing
/// kernel, stride, padding, startCol, startRow (4 bits each from bit 0),
/// windowStart [35:26], windowCount [45:36] and tile rows [50:46];
/// `input_groups` is op1_col.
pub fn cycles(iter: u64, rs2: u64, input_groups: u64) -> u64 {
    let kernel = rs2 & 0xf;
    let stride = (rs2 >> 4) & 0xf;
    let padding = (rs2 >> 8) & 0xf;
    let start_col = (rs2 >> 12) & 0xf;
    let start_row = (rs2 >> 16) & 0xf;
    let window_start = (rs2 >> 26) & 0x3ff;
    let window_count = (rs2 >> 36) & 0x3ff;
    let tile_rows = (rs2 >> 46) & 0x1f;
    let output_cols = (iter + 2 * padding - kernel - start_col) / stride + 1;
    let row = start_row + window_start / output_cols * stride;
    let col = start_col + window_start % output_cols * stride;
    let first_read = row >= padding && row < padding + iter && col >= padding && col < padding + iter;
    let rows = kernel * kernel * input_groups * window_count.div_ceil(tile_rows) * tile_rows;
    71 + u64::from(first_read) + 2 * rows
}
