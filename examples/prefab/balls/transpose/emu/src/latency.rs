// The cycles of one TRANSPOSE instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle. Delays outside the Ball belong to the bank and bus model, not
// here. From Transpose.scala, the elements are walked source-row first (srcR
// fastest) into destination beats of E = row bytes / element bytes elements,
// iter * cols beats in all:
//   1          idle -> sRead
//   iter > 1:
//   3 / elem   sRead: needRead (srcR moved off the cached row) raises the read
//              request; its response the next cycle fills the cache; the next
//              cycle consumes the element (cached && !needRead) and steps
//              srcR, so the next element's request comes after it
//   iter = 1:  srcR stays 0, the cached row only changes with the group: per
//              beat one read (2), then E elements one per cycle
//   2 / beat   sWrite: the write request, then its acknowledgement -> sRead
//   complete   the response is valid the cycle after the last
//              acknowledgement.
// The UVM tests of the RTL check it exactly.

/// One TRANSPOSE as the Ball receives it: `iter` source rows of `cols` groups,
/// elements of `elem_bits` (rs2[7:0]) in rows of `row_bytes`.
pub fn cycles(iter: u64, elem_bits: u64, cols: u64, row_bytes: u64) -> u64 {
    let elements = row_bytes * 8 / elem_bits;
    if iter == 1 {
        cols * (2 + elements + 2) + 1
    } else {
        iter * cols * (3 * elements + 2) + 1
    }
}
