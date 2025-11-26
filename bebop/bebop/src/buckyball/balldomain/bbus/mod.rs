/// BBus - Ball bus with registration pattern
use crate::builtin::{Module, Wire, Ball};
use crate::buckyball::frontend::GlobalRsComplete;
use crate::buckyball::prototype::*;
use super::rs::BallIssue;

const NUM_BALLS: usize = 5;

/// BBus manages all registered balls using Ball trait
pub struct BBus {
  name: String,
  balls: Vec<Box<dyn Ball>>,
  pub cmd_reqs: [Wire<BallIssue>; NUM_BALLS],
  pub cmd_resps: [Wire<GlobalRsComplete>; NUM_BALLS],
}

impl BBus {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      balls: vec![
        Box::new(VecBall::new(0)),
        Box::new(MatrixBall::new(1)),
        Box::new(Im2colBall::new(2)),
        Box::new(TransposeBall::new(3)),
        Box::new(ReluBall::new(4)),
      ],
      cmd_reqs: Default::default(),
      cmd_resps: Default::default(),
    }
  }
}

impl Module for BBus {
  fn run(&mut self) {
    // Route commands to balls with raw xs1/xs2 for ball-specific ISA decoding
    for (bid, ball) in self.balls.iter_mut().enumerate() {
      if self.cmd_reqs[bid].valid {
        let issue = &self.cmd_reqs[bid].value;
        let blink = ball.blink_mut();
        blink.cmd_req.valid = true;
        blink.cmd_req.rob_id = issue.rob_id as u32;
        blink.cmd_req.bid = bid as u8;
        blink.cmd_req.funct = issue.funct;
        blink.cmd_req.xs1 = issue.xs1;
        blink.cmd_req.xs2 = issue.xs2;
      }
    }

    // Tick all balls
    for ball in &mut self.balls {
      ball.tick();
    }

    // Collect responses
    for i in 0..NUM_BALLS {
      self.cmd_resps[i].clear();
    }
    for ball in &self.balls {
      let blink = ball.blink();
      if blink.cmd_resp.valid {
        let bid = ball.ball_id() as usize;
        self.cmd_resps[bid].set(GlobalRsComplete {
          rob_id: blink.cmd_resp.rob_id as usize,
          data: 0,
        });
      }
    }
  }

  fn reset(&mut self) {
    for ball in &mut self.balls {
      ball.reset();
    }
    self.cmd_reqs = Default::default();
    self.cmd_resps = Default::default();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
