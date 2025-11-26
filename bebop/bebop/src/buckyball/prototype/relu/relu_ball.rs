/// ReluBall - ReLU activation accelerator
use crate::builtin::{Module, Wire};
use super::isa::ReluCmd;
use super::compute::ReluCompute;
use super::runner::run_relu;
use super::super::common::MemoryInterface;

pub struct ReluBall {
    name: String,
    pub cmd_req: Wire<ReluCmd>,
    pub cmd_resp: Wire<u32>,
    pub mem: MemoryInterface,
    compute: ReluCompute,
    raddr: u32,
    waddr: u32,
    idle: bool,
}

impl ReluBall {
    pub fn new(bid: u8) -> Self {
        Self {
            name: format!("relu_ball_{}", bid),
            cmd_req: Wire::default(),
            cmd_resp: Wire::default(),
            mem: MemoryInterface::default(),
            compute: ReluCompute::new(),
            raddr: 0,
            waddr: 0,
            idle: true,
        }
    }
}

impl Module for ReluBall {
    fn run(&mut self) {
        if self.idle && self.cmd_req.valid {
            let cmd = &self.cmd_req.value;
            self.raddr = cmd.op1_addr;
            self.waddr = cmd.dst_addr;
            self.compute.reset();
            self.idle = false;
        }

        if !self.idle {
            let done = run_relu(
                &mut self.mem,
                &mut self.compute,
                self.raddr,
                self.waddr,
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
