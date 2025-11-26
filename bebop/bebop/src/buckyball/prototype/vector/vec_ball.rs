/// VecBall - Vector processing accelerator
/// Performs vector operations (add, mul, etc.)
use crate::builtin::{Module, Wire};
use super::isa::VecCmd;

pub struct VecBall {
    name: String,
    pub cmd_req: Wire<VecCmd>,
    pub cmd_resp: Wire<u32>,

    state: State,
    cycle_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    Loading,
    Computing,
    Storing,
    Complete,
}

impl VecBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("vec_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            state: State::Idle,
            cycle_count: 0,
        }
    }
}

impl Module for VecBall {
    fn run(&mut self) {
        match self.state {
            State::Idle => {
                if self.cmd_req.valid {
                    let cmd = &self.cmd_req.value;
                    println!("  [VecBall] Starting vector op: op1=0x{:x}, op2=0x{:x}, dst=0x{:x}, iter={}",
                        cmd.op1_addr, cmd.op2_addr, cmd.dst_addr, cmd.iter);
                    self.state = State::Loading;
                    self.cycle_count = 0;
                    self.cmd_resp.clear();
                }
            },
            State::Loading => {
                // Load operands from scratchpad
                self.cycle_count += 1;
                if self.cycle_count >= 2 {
                    self.state = State::Computing;
                    self.cycle_count = 0;
                }
            },
            State::Computing => {
                // Perform vector computation
                self.cycle_count += 1;
                if self.cycle_count >= 1 {
                    self.state = State::Storing;
                    self.cycle_count = 0;
                }
            },
            State::Storing => {
                // Store result to accumulator
                self.cycle_count += 1;
                if self.cycle_count >= 2 {
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
