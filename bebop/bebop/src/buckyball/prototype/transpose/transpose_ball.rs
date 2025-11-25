/// TransposeBall - Matrix transpose accelerator
use crate::buckyball::builtin::{Module, Wire};
use super::isa::TransposeCmd;

pub struct TransposeBall {
    name: String,
    pub cmd_req: Wire<TransposeCmd>,
    pub cmd_resp: Wire<u32>,
    busy: bool,
}

impl TransposeBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("transpose_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            busy: false,
        }
    }
}

impl Module for TransposeBall {
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
