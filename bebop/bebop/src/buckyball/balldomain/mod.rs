/// BallDomain - Ball computation domain
pub mod decoder;
pub mod rs;
pub mod bbus;

use crate::builtin::{Module, Wire};
use crate::buckyball::frontend::{GlobalRsComplete, GlobalRsIssue};

pub struct BallDomain {
  name: String,

  decoder: decoder::DomainDecoder,
  rs: rs::ReservationStation,
  bbus: bbus::BBus,

  pub global_issue_i: Wire<GlobalRsIssue>,
  pub global_complete_o: Wire<GlobalRsComplete>,
}

impl BallDomain {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      decoder: decoder::DomainDecoder::new("ball_decoder"),
      rs: rs::ReservationStation::new("ball_rs"),
      bbus: bbus::BBus::new("bbus"),
      global_issue_i: Wire::default(),
      global_complete_o: Wire::default(),
    }
  }
}

impl Module for BallDomain {
  fn run(&mut self) {
    // Run from back to front

    // 1. Run BBus (execute ball operations)
    self.bbus.run();

    // 2. Run RS (issue to BBus, forward completion)
    self.rs.run();

    // 3. Run Decoder (decode incoming instruction)
    self.decoder.run();

    // 4. Wire updates

    // Global RS -> Decoder
    self.decoder.input.set(self.global_issue_i.value.cmd.clone());

    // Decoder -> RS (with rob_id from Global RS)
    self.rs.decode_input.set(self.global_issue_i.value.clone());

    // RS -> BBus (multi-channel)
    for i in 0..5 {
      self.bbus.cmd_reqs[i] = self.rs.ball_issues[i].clone();
    }

    // BBus -> RS (completion)
    for i in 0..5 {
      self.rs.ball_completes[i] = self.bbus.cmd_resps[i].clone();
    }

    // RS -> Global RS (completion)
    self.global_complete_o = self.rs.complete_output.clone();
  }

  fn reset(&mut self) {
    self.decoder.reset();
    self.rs.reset();
    self.bbus.reset();
    self.global_issue_i = Wire::default();
    self.global_complete_o = Wire::default();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
