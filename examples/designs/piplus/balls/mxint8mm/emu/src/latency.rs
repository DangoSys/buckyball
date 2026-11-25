// The cycles of one MXINT8MM instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle (the port free the cycle after), the response accepted at once.
// From Mxint8mmBall.scala and Loader.scala, block after block of the tiles,
// row-major, each into slot j % 2:
//   Load: from its first cycle (the loader done with the block before and
//   its slot empty) a block issues one read a cycle on each port, answered a
//   cycle later; `loaded` follows the last answer, the slot is full the next:
//   R + 2 cycles, R the longer port's reads: per A row and per B column 3
//   (two code rows and the exponent row) in a block 0 mod 16, else 2; port A
//   first reads the old C (rows * groups bank rows, accumulate) and the bias
//   (groups, bias) in the tile's first block.
//   MAC: once full and the block before done, `steps` cycles a row. A tile's
//   last block starts once the store of the tile before has ended (storing
//   low); its first block (when not also its last) starts row r once that
//   store has written row r (storeRow > r) or ended. A row converts the
//   cycle after its last MAC cycle, adds the next, is added the next; the
//   slot is empty the cycle after its last row converts.
//   Store: the last block's first MAC cycle starts it; a write of four
//   columns once its row is added, one every 2 cycles; storeRow passes a row
//   the cycle after its last write, storing falls the cycle after the last.
//   Done: once storing, the writes in flight, conversion and addition are
//   over and the slots are empty; the response the cycle after.
// The UVM tests of the RTL check it exactly.

/// One MXINT8MM instruction as the Ball receives it (K = `iter`, rs2) on an
/// array of `rows` x `cols` columns taking `steps` MAC cycles a row.
pub fn cycles(iter: u64, rs2: u64, rows: u64, cols: u64, steps: u64) -> u64 {
    let m = rs2 & 0x3ff;
    let n = (rs2 >> 10) & 0x3ff;
    let accumulate = (rs2 >> 62) & 1 == 1;
    let bias = rs2 >> 63 == 1;
    let blocks = iter / 32;
    // The first cycle the loader may start a block, each slot is empty, the
    // MAC may take a block and the write port takes a write.
    let mut loader = 1;
    let mut empty = [0u64; 2];
    let mut mac = 0;
    let mut port = 0;
    // The store of the tile before: the cycle it passed each row, its end.
    let mut stored: Vec<u64> = Vec::new();
    let mut store_end = 0;
    let mut done = 0;
    let mut slot = 0;
    for m0 in (0..m).step_by(rows as usize) {
        for n0 in (0..n).step_by(cols as usize) {
            let active_rows = rows.min(m - m0);
            let active_cols = cols.min(n - n0);
            let groups = active_cols / 4;
            for block in 0..blocks {
                let parts = if block % 16 == 0 { 3 } else { 2 };
                let mut reads_a = active_rows * parts;
                if block == 0 && accumulate {
                    reads_a += active_rows * groups;
                }
                if block == 0 && bias {
                    reads_a += groups;
                }
                let full = loader.max(empty[slot]) + reads_a.max(active_cols * parts) + 2;
                loader = full;
                let first = block == 0;
                let last = block == blocks - 1;
                let mut t = full.max(mac);
                if last {
                    t = t.max(store_end);
                }
                // The last MAC cycle of each row.
                let mut ends = Vec::new();
                for r in 0..active_rows as usize {
                    if first && !last {
                        t = t.max(stored.get(r).map_or(store_end, |&at| at.min(store_end)));
                    }
                    t += steps;
                    ends.push(t - 1);
                }
                mac = t;
                empty[slot] = t + 1;
                if last {
                    stored.clear();
                    for end in &ends {
                        for _ in 0..groups {
                            port = port.max(end + 3) + 2;
                        }
                        stored.push(port - 1);
                    }
                    store_end = port - 1;
                    done = port.max(t + 2);
                }
                slot ^= 1;
            }
        }
    }
    done + 1
}
