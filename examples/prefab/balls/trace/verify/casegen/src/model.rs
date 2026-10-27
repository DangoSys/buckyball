// TraceBall BDB_COUNTER contract. rs2[3:0] is the subcommand, rs2[7:4] the
// counter and rs2[63:8] the payload; the Ball touches no bank. Each command
// emits one trace record (subcmd, counter, tag, elapsed, cycle):
//   START (0): records cycle as the counter's start and the payload as its
//              tag; the record carries the payload and elapsed 0.
//   STOP  (1): elapsed = cycle - start; the payload is ignored.
//   READ  (2): like STOP but the counter keeps running.

pub const CTR_START: u32 = 0;
pub const CTR_STOP: u32 = 1;
pub const CTR_READ: u32 = 2;
pub const NUM_COUNTERS: usize = 16;
pub const PAYLOAD_MASK: u64 = (1 << 56) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    pub subcmd: u32,
    pub ctr: u32,
    pub payload: u64,
}

impl Command {
    pub fn rs2(&self) -> u64 {
        if self.subcmd > CTR_READ || self.ctr as usize >= NUM_COUNTERS || self.payload > PAYLOAD_MASK {
            panic!("trace: invalid command {self:?}");
        }
        (self.payload << 8) | (u64::from(self.ctr) << 4) | u64::from(self.subcmd)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub subcmd: u32,
    pub ctr: u32,
    pub tag: u64,
    pub elapsed: u64,
    pub cycle: u64,
}

#[derive(Default)]
pub struct Counters {
    start: [Option<(u64, u64)>; NUM_COUNTERS],
    last_cycle: Option<u64>,
}

impl Counters {
    // The expected record of `cmd` given the cycle the Ball reported for it.
    pub fn expect(&mut self, cmd: &Command, cycle: u64) -> Result<Record, String> {
        if let Some(last) = self.last_cycle {
            if cycle <= last {
                return Err(format!("cycle {cycle} does not advance past {last}"));
            }
        }
        self.last_cycle = Some(cycle);
        let ctr = cmd.ctr as usize;
        if cmd.subcmd == CTR_START {
            self.start[ctr] = Some((cycle, cmd.payload));
            return Ok(Record { subcmd: CTR_START, ctr: cmd.ctr, tag: cmd.payload, elapsed: 0, cycle });
        }
        let (start, tag) = match self.start[ctr] {
            Some(entry) => entry,
            None => return Err(format!("counter {ctr} used before START")),
        };
        if cmd.subcmd == CTR_STOP {
            self.start[ctr] = None;
        }
        Ok(Record { subcmd: cmd.subcmd, ctr: cmd.ctr, tag, elapsed: cycle - start, cycle })
    }
}
