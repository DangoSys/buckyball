/// Im2colBall - Image to column transformation accelerator
use crate::builtin::ball::{Ball, Blink};
use super::decode;
use super::compute::Im2colCompute;
use super::runner::run_im2col;
use super::super::common::{NUM_SP_BANKS, NUM_ACC_BANKS};

pub struct Im2colBall {
  bid: u8,
  blink: Blink,
  compute: Im2colCompute,
  src_addr: u32,
  dst_addr: u32,
  idle: bool,
}

impl Im2colBall {
  pub fn new(bid: u8) -> Self {
    Self {
      bid,
      blink: Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS),
      compute: Im2colCompute::new(),
      src_addr: 0, dst_addr: 0,
      idle: true,
    }
  }
}

impl Ball for Im2colBall {
  fn ball_id(&self) -> u8 { self.bid }
  fn blink(&self) -> &Blink { &self.blink }
  fn blink_mut(&mut self) -> &mut Blink { &mut self.blink }

  fn tick(&mut self) {
    self.blink.clear_requests();
    if self.idle && self.blink.cmd_req.valid {
      let cmd = decode::decode(&self.blink.cmd_req);
      self.src_addr = cmd.op1_addr;
      self.dst_addr = cmd.op2_addr;
      self.compute.reset();
      self.idle = false;
    }
    if !self.idle {
      let done = run_im2col(&mut self.blink, &mut self.compute,
        self.src_addr, self.dst_addr);
      if done { self.idle = true; }
    }
  }

  fn reset(&mut self) {
    self.blink = Blink::new(NUM_SP_BANKS, NUM_ACC_BANKS);
    self.compute.reset();
    self.idle = true;
  }
}
