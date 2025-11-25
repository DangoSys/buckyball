/// Im2colBall - Image to column transformation accelerator
use crate::buckyball::builtin::{Module, Wire};
use super::isa::Im2colCmd;

pub struct Im2colBall {
    name: String,
    pub cmd_req: Wire<Im2colCmd>,
    pub cmd_resp: Wire<u32>,
    busy: bool,
}

impl Im2colBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("im2col_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            busy: false,
        }
    }
}

impl Module for Im2colBall {
    fn run(&mut self) {
        if !self.busy && self.cmd_req.valid {
            self.busy = true;
            self.cmd_resp.clear();
        } else if self.busy {
            self.cmd_resp.set(0);
            self.busy = false;
        } else {
            self.cmd_resp.clear();
        }
    }

    fn reset(&mut self) {
        self.cmd_req = Wire::default();
        self.cmd_resp = Wire::default();
        self.busy = false;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
