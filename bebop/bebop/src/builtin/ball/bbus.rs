/// BBus - Ball bus that manages registered balls
/// Mirrors RTL: framework/bbus/bbus.scala
use super::traits::Ball;
use super::blink::*;
use crate::builtin::Module;

pub struct BBus {
  name: String,
  balls: Vec<Box<dyn Ball>>,
}

impl BBus {
  pub fn new(balls: Vec<Box<dyn Ball>>) -> Self {
    Self {
      name: "bbus".to_string(),
      balls,
    }
  }

  pub fn num_balls(&self) -> usize {
    self.balls.len()
  }

  pub fn get_ball(&self, bid: u8) -> Option<&Box<dyn Ball>> {
    self.balls.iter().find(|b| b.ball_id() == bid)
  }

  pub fn get_ball_mut(&mut self, bid: u8) -> Option<&mut Box<dyn Ball>> {
    self.balls.iter_mut().find(|b| b.ball_id() == bid)
  }

  /// Route command to target ball
  pub fn route_cmd(&mut self, cmd: &BallCmdReq) {
    if let Some(ball) = self.get_ball_mut(cmd.bid) {
      let blink = ball.blink_mut();
      blink.cmd_req = cmd.clone();
    }
  }

  /// Collect responses from all balls
  pub fn collect_responses(&self) -> Vec<BallCmdResp> {
    self.balls.iter()
      .map(|b| b.blink().cmd_resp.clone())
      .filter(|r| r.valid)
      .collect()
  }

  /// Get SRAM read requests from all balls
  pub fn collect_sram_read_reqs(&self) -> Vec<(u8, &SramReadReq)> {
    let mut reqs = Vec::new();
    for ball in &self.balls {
      for req in &ball.blink().sram_read_req {
        if req.valid {
          reqs.push((ball.ball_id(), req));
        }
      }
    }
    reqs
  }

  /// Dispatch SRAM read response to ball
  pub fn dispatch_sram_read_resp(&mut self, bid: u8, bank: usize, resp: SramReadResp) {
    if let Some(ball) = self.get_ball_mut(bid) {
      if bank < ball.blink().sram_read_resp.len() {
        ball.blink_mut().sram_read_resp[bank] = resp;
      }
    }
  }
}

impl Module for BBus {
  fn run(&mut self) {
    for ball in &mut self.balls {
      ball.tick();
    }
  }

  fn reset(&mut self) {
    for ball in &mut self.balls {
      ball.reset();
    }
  }

  fn name(&self) -> &str {
    &self.name
  }
}
