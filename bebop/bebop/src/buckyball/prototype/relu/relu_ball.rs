/// ReluBall - ReLU activation accelerator
use crate::builtin::ball::{Ball, Blink};
use super::decode;
use super::compute::ReluCompute;
use super::runner::run_relu;
use super::super::common::{NUM_SP_BANKS, NUM_ACC_BANKS};

pub struct ReluBall {
  bid: u8,
  blink: Blink,
  compute: ReluCompute,
  raddr: u32,
  waddr: u32,
  idle: bool,
}

impl ReluBall {
  pub fn new(bid: u8) -> Self {
    Self {
      bid,
      blink: Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS),
      compute: ReluCompute::new(),
      raddr: 0, waddr: 0,
      idle: true,
    }
  }
}

impl Ball for ReluBall {
  fn ball_id(&self) -> u8 { self.bid }
  fn blink(&self) -> &Blink { &self.blink }
  fn blink_mut(&mut self) -> &mut Blink { &mut self.blink }

  fn tick(&mut self) {
    self.blink.clear_requests();

    if self.idle && self.blink.cmd_req.valid {
      let cmd = decode::decode(&self.blink.cmd_req);
      self.raddr = cmd.op1_addr;
      self.waddr = cmd.dst_addr;
      self.compute.reset();
      self.idle = false;
    }

    if !self.idle {
      let done = run_relu(&mut self.blink, &mut self.compute, self.raddr, self.waddr);
      if done { self.idle = true; }
    }
  }

  fn reset(&mut self) {
    self.blink = Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS);
    self.compute.reset();
    self.idle = true;
  }
}
