/// MatrixBall - Matrix computation accelerator
use crate::buckyball::builtin::{Module, Wire};
use super::isa::MatrixCmd;

pub struct MatrixBall {
    name: String,
    pub cmd_req: Wire<MatrixCmd>,
    pub cmd_resp: Wire<u32>,
    busy: bool,
}

impl MatrixBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("matrix_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            busy: false,
        }
    }
}

impl Module for MatrixBall {
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
