//===- 04_bdb_counter.rs - BDB_COUNTER instruction -------------------------===//
//
// rs2[3:0] is the subcommand, rs2[7:4] the counter and rs2[63:8] the payload;
// the Ball touches no bank. Each command emits one counter trace record:
//   START (0): the issue cycle becomes the counter's start, the payload its tag.
//   STOP  (1): elapsed = cycle - start of a started counter, which is freed.
//   READ  (2): like STOP but the counter keeps running.
// Subcommands 3..15 and STOP/READ of a counter that is not started are illegal.
//
//===----------------------------------------------------------------------===//

use super::instruction::{BallModel, ExecContext};

const CTR_START: u32 = 0;
const CTR_STOP: u32 = 1;
const CTR_READ: u32 = 2;
const NUM_COUNTERS: usize = 16;

/// (start cycle, tag) of each started counter.
#[derive(Default)]
struct Counters([Option<(u64, u64)>; NUM_COUNTERS]);

pub struct BdbCounter;

impl BallModel for BdbCounter {
    fn exec(_xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let subcmd = (xs2 & 0xf) as u32;
        let ctr = ((xs2 >> 4) & 0xf) as u32;
        let payload = xs2 >> 8;
        assert!(subcmd <= CTR_READ, "bdb_counter: subcmd {subcmd} is illegal");
        let cycle = ctx.cycle;
        let state = ctx.state::<Counters>();
        let mut counters = state.lock().expect("bdb_counter state poisoned");
        let slot = &mut counters.0[ctr as usize];
        if subcmd == CTR_START {
            *slot = Some((cycle, payload));
            bebop_bemu::thread::trace::ctrace(CTR_START, ctr, payload, 0, cycle);
            return 0;
        }
        let (start, tag) = slot.unwrap_or_else(|| panic!("bdb_counter: counter {ctr} is not started"));
        if subcmd == CTR_STOP {
            *slot = None;
        }
        bebop_bemu::thread::trace::ctrace(subcmd, ctr, tag, cycle - start, cycle);
        0
    }

    fn latency(_xs1: u64, _xs2: u64) -> u64 {
        1
    }
}
