/// TransposeBall - Matrix transpose accelerator
/// Reads matrix from scratchpad, transposes it, writes back
use crate::builtin::{Module, Wire};
use super::isa::TransposeCmd;

pub struct TransposeBall {
    name: String,
    pub cmd_req: Wire<TransposeCmd>,
    pub cmd_resp: Wire<u32>,

    state: State,
    cycle_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    Reading,
    Transposing,
    Writing,
    Complete,
}

impl TransposeBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("transpose_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            state: State::Idle,
            cycle_count: 0,
        }
    }
}

impl Module for TransposeBall {
    fn run(&mut self) {
        match self.state {
            State::Idle => {
                if self.cmd_req.valid {
                    let cmd = &self.cmd_req.value;
                    println!("  [TransposeBall] Starting transpose: op1_addr=0x{:x}, op2_addr=0x{:x}, iter={}",
                        cmd.op1_addr, cmd.op2_addr, cmd.iter);
                    self.state = State::Reading;
                    self.cycle_count = 0;
                    self.cmd_resp.clear();
                }
            },
            State::Reading => {
                self.cycle_count += 1;
                if self.cycle_count >= 3 {
                    self.state = State::Transposing;
                    self.cycle_count = 0;
                }
            },
            State::Transposing => {
                // Simulate matrix transpose
                self.cycle_count += 1;
                if self.cycle_count >= 2 {
                    self.state = State::Writing;
                    self.cycle_count = 0;
                }
            },
            State::Writing => {
                self.cycle_count += 1;
                if self.cycle_count >= 3 {
                    self.state = State::Complete;
                }
            },
            State::Complete => {
                self.cmd_resp.set(0);
                self.state = State::Idle;
                self.cycle_count = 0;
            },
        }
    }

    fn reset(&mut self) {
        self.cmd_req = Wire::default();
        self.cmd_resp = Wire::default();
        self.state = State::Idle;
        self.cycle_count = 0;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
