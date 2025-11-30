/// BBus - Ball bus with registration pattern
use crate::builtin::{Module, Ball};
use crate::buckyball::prototype::*;

/// BBus manages all registered balls using Ball trait
pub struct BBus {
  name: String,
  balls: Vec<Box<dyn Ball>>,
  pending: Option<(u64, u64, u64)>, // (funct, xs1, xs2)
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
      pending: None,
    }
  }

  /// 发送指令
  pub fn issue(&mut self, funct: u64, xs1: u64, xs2: u64) {
    self.pending = Some((funct, xs1, xs2));
  }
}

impl Module for BBus {
  fn tick(&mut self) {
    // Route pending command to appropriate ball
    if let Some((funct, xs1, xs2)) = self.pending.take() {
      // Determine which ball based on funct
      let ball_id = match funct {
        32..=33 => 0, // VecBall
        34..=35 => 1, // MatrixBall
        36..=37 => 2, // Im2colBall
        38..=39 => 3, // TransposeBall
        40..=42 => 4, // ReluBall
        _ => return,
      };

      if ball_id < self.balls.len() {
        let ball = &mut self.balls[ball_id];
        let blink = ball.blink_mut();
        blink.cmd_req.valid = true;
        blink.cmd_req.rob_id = 0;
        blink.cmd_req.bid = ball_id as u8;
        blink.cmd_req.funct = funct as u8;
        blink.cmd_req.xs1 = xs1;
        blink.cmd_req.xs2 = xs2;
      }
    }

    // Tick all balls
    for ball in &mut self.balls {
      ball.tick();
    }
  }

  fn name(&self) -> &str {
    &self.name
  }
}
