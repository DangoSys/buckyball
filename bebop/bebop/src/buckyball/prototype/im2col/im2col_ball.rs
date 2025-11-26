/// Im2colBall - Image to column transformation accelerator
/// Converts image patches to columns for convolution
use crate::builtin::{Module, Wire};
use super::isa::Im2colCmd;

pub struct Im2colBall {
    name: String,
    pub cmd_req: Wire<Im2colCmd>,
    pub cmd_resp: Wire<u32>,

    state: State,
    cycle_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Idle,
    Reading,
    Converting,
    Writing,
    Complete,
}

impl Im2colBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("im2col_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            state: State::Idle,
            cycle_count: 0,
        }
    }
}

impl Module for Im2colBall {
    fn run(&mut self) {
        match self.state {
            State::Idle => {
                if self.cmd_req.valid {
                    let cmd = &self.cmd_req.value;
                    println!("  [Im2colBall] Starting im2col: op1_addr=0x{:x}, dst_addr=0x{:x}, iter={}",
                        cmd.op1_addr, cmd.dst_addr, cmd.iter);
                    self.state = State::Reading;
                    self.cycle_count = 0;
                    self.cmd_resp.clear();
                }
            },
            State::Reading => {
                self.cycle_count += 1;
                if self.cycle_count >= 4 {
                    self.state = State::Converting;
                    self.cycle_count = 0;
                }
            },
            State::Converting => {
                // Simulate im2col conversion
                self.cycle_count += 1;
                if self.cycle_count >= 3 {
                    self.state = State::Writing;
                    self.cycle_count = 0;
                }
            },
            State::Writing => {
                self.cycle_count += 1;
                if self.cycle_count >= 4 {
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
