/// MatrixBall - Matrix multiplication accelerator
use crate::builtin::{Module, Wire};
use super::isa::MatrixCmd;
use super::compute::MatrixCompute;
use super::runner::run_matrix;
use super::super::common::MemoryInterface;

pub struct MatrixBall {
    name: String,
    pub cmd_req: Wire<MatrixCmd>,
    pub cmd_resp: Wire<u32>,
    pub mem: MemoryInterface,
    compute: MatrixCompute,
    op1_addr: u32,
    op2_addr: u32,
    dst_addr: u32,
    idle: bool,
}

impl MatrixBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("matrix_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            mem: MemoryInterface::default(),
            compute: MatrixCompute::new(),
            op1_addr: 0,
            op2_addr: 0,
            dst_addr: 0,
            idle: true,
        }
    }
}

impl Module for MatrixBall {
    fn run(&mut self) {
        if self.idle && self.cmd_req.valid {
            let cmd = &self.cmd_req.value;
            self.op1_addr = cmd.op1_addr;
            self.op2_addr = cmd.op2_addr;
            self.dst_addr = cmd.dst_addr;
            self.compute.reset();
            self.idle = false;
        }

        if !self.idle {
            let done = run_matrix(
                &mut self.mem,
                &mut self.compute,
                self.op1_addr,
                self.op2_addr,
                self.dst_addr,
                &mut self.cmd_resp
            );
            if done {
                self.idle = true;
            }
        }
    }

    fn reset(&mut self) {
        self.cmd_req = Wire::default();
        self.cmd_resp = Wire::default();
        self.mem = MemoryInterface::default();
        self.compute.reset();
        self.idle = true;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
