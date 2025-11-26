/// MatrixBall - Matrix multiplication accelerator
use crate::builtin::ball::{Ball, Blink};
use super::decode;
use super::compute::MatrixCompute;
use super::runner::run_matrix;
use super::super::common::{NUM_SP_BANKS, NUM_ACC_BANKS};

pub struct MatrixBall {
  bid: u8,
  blink: Blink,
  compute: MatrixCompute,
  op1_addr: u32,
  op2_addr: u32,
  dst_addr: u32,
  idle: bool,
}

impl MatrixBall {
  pub fn new(bid: u8) -> Self {
    Self {
      bid,
      blink: Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS),
      compute: MatrixCompute::new(),
      op1_addr: 0, op2_addr: 0, dst_addr: 0,
      idle: true,
    }
  }
}

impl Ball for MatrixBall {
  fn ball_id(&self) -> u8 { self.bid }
  fn blink(&self) -> &Blink { &self.blink }
  fn blink_mut(&mut self) -> &mut Blink { &mut self.blink }

  fn tick(&mut self) {
    self.blink.clear_requests();
    if self.idle && self.blink.cmd_req.valid {
      let cmd = decode::decode(&self.blink.cmd_req);
      self.op1_addr = cmd.op1_addr;
      self.op2_addr = cmd.op2_addr;
      self.dst_addr = cmd.dst_addr;
      self.compute.reset();
      self.idle = false;
    }
    if !self.idle {
      let done = run_matrix(&mut self.blink, &mut self.compute,
        self.op1_addr, self.op2_addr, self.dst_addr);
      if done { self.idle = true; }
    }
  }

  fn reset(&mut self) {
    self.blink = Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS);
    self.compute.reset();
    self.idle = true;
  }
}
