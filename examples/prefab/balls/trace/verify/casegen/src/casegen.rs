use crate::model::{Command, CTR_READ, CTR_START, CTR_STOP, NUM_COUNTERS, PAYLOAD_MASK};

fn start(ctr: u32, payload: u64) -> Command {
    Command { subcmd: CTR_START, ctr, payload }
}

fn stop(ctr: u32, payload: u64) -> Command {
    Command { subcmd: CTR_STOP, ctr, payload }
}

fn read(ctr: u32, payload: u64) -> Command {
    Command { subcmd: CTR_READ, ctr, payload }
}

pub fn gen_case(index: u32) -> Vec<Command> {
    match index {
        // One measured region with a read inside.
        0 => vec![start(0, 0xa001), read(0, 0), stop(0, 0)],
        // Nested and sibling regions, as in the README example.
        1 => vec![
            start(0, 0xb001),
            start(1, 0xb002),
            stop(1, 0),
            start(2, 0xb003),
            read(0, 0),
            stop(2, 0),
            stop(0, 0),
        ],
        // Every counter, full-width tags; STOP and READ payloads are ignored.
        2 => {
            let mut cmds = Vec::new();
            for ctr in 0..NUM_COUNTERS as u32 {
                let tag = if ctr % 2 == 0 {
                    PAYLOAD_MASK ^ (u64::from(ctr) << 40)
                } else {
                    0x0055_aa55_aa55_aa55 ^ u64::from(ctr)
                };
                cmds.push(start(ctr, tag));
            }
            for ctr in 0..NUM_COUNTERS as u32 {
                cmds.push(read(ctr, PAYLOAD_MASK));
            }
            for ctr in (0..NUM_COUNTERS as u32).rev() {
                cmds.push(stop(ctr, PAYLOAD_MASK ^ u64::from(ctr)));
            }
            cmds
        }
        // A restarted counter measures from the new START with the new tag;
        // a zero tag is a valid tag.
        3 => vec![
            start(5, 0),
            stop(5, 0),
            start(5, 0x2222),
            read(5, 0),
            read(5, 0),
            stop(5, 0),
        ],
        _ => panic!("trace: unsupported case {index}"),
    }
}
