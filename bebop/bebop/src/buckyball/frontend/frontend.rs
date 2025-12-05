use crate::builtin::{Sim, EventQueue};
use crate::buckyball::top::{MemCmd, CmdResponse};
use super::{Decoder, Rob, Rs, DomainScheduler};
use std::sync::mpsc::Sender;
use crate::{log_forward, log_error, log_tpc};

pub struct Frontend {
  decoder: Decoder,
  rob: Rob,
  rs: Rs,
  domain_scheduler: DomainScheduler,
  pub event_queue: EventQueue<Frontend>,

  // Channel to send memory commands to MemDomain
  mem_cmd_tx: Option<Sender<MemCmd>>,

  // Channel to send command responses
  cmd_response_tx: Option<Sender<CmdResponse>>,

  pub funct: u32,
  pub xs1: u64,
  pub xs2: u64,
}

impl Frontend {
  pub fn new() -> Self {
    Self {
      decoder: Decoder::new(),
      rob: Rob::new(),
      rs: Rs::new(),
      domain_scheduler: DomainScheduler::new(),
      event_queue: EventQueue::new(),
      mem_cmd_tx: None,
      cmd_response_tx: None,

      funct: 0,
      xs1: 0,
      xs2: 0,
    }
  }

  /// Set the memory command sender channel
  pub fn set_mem_cmd_sender(&mut self, sender: Sender<MemCmd>) {
    self.mem_cmd_tx = Some(sender);
  }

  /// Set the command response sender channel
  pub fn set_cmd_response_sender(&mut self, sender: Sender<CmdResponse>) {
    self.cmd_response_tx = Some(sender);
  }

  pub fn rocc_cmd(&mut self) {
    let funct = self.funct;
    let xs1 = self.xs1;
    let xs2 = self.xs2;

    self.event_queue.push("RoccCmd", move |frontend: &mut Frontend| {
      frontend.decoder.decode_cmd(funct, xs1, xs2);
    });
    log_forward!("Inst decode!");
    self.enter_rob();
  }

  pub fn enter_rob(&mut self) {
    let funct = self.decoder.funct;
    let xs1 = self.decoder.xs1;
    let xs2 = self.decoder.xs2;
    let is_fence = self.decoder.is_fence;

    self.event_queue.push("EnterRob", move |frontend: &mut Frontend| {
      frontend.rob.enter_rob(funct, xs1, xs2, is_fence);
    });
    log_forward!("Inst enter rob!");

    // Fence instructions don't go through dispatch/issue
    if !is_fence {
      // Normal instructions: send cmd_response immediately after entering ROB
      if let Some(ref tx) = self.cmd_response_tx {
        let cmd_response = CmdResponse { result: 0 };
        if let Err(e) = tx.send(cmd_response) {
          log_error!("Failed to send CmdResponse: {}", e);
        } else {
          log_tpc!("Normal instruction entered ROB, sending cmd_response immediately");
        }
      }
    } else {
      // For fence, we'll send response when ROB is empty (checked in backward)
      log_forward!("Fence instruction - will wait for ROB empty");
    }

    self.dispatch_cmd();

  }

  pub fn dispatch_cmd(&mut self) {
    let funct = self.rob.funct;
    let xs1 = self.rob.xs1;
    let xs2 = self.rob.xs2;
    let rob_id = self.rob.rob_id;

    self.event_queue.push("DispatchCmd", move |frontend: &mut Frontend| {
      frontend.domain_scheduler.dispatch_cmd(funct, xs1, xs2, rob_id);
    });
    log_forward!("Inst dispatch!");
    self.issue_cmd();
  }

  pub fn issue_cmd(&mut self) {
    let funct = self.domain_scheduler.funct;
    let xs1 = self.domain_scheduler.xs1;
    let xs2 = self.domain_scheduler.xs2;
    let rob_id = self.domain_scheduler.rob_id;
    let domain_id = self.domain_scheduler.domain_id;

    self.event_queue.push("IssueCmd", move |frontend: &mut Frontend| {
      frontend.rs.issue_cmd(funct, xs1, xs2, rob_id, domain_id);

      // Send memory command to MemDomain if it's a memory operation (funct=24 or 25)
      if domain_id == 1 {
        if let Some(ref tx) = frontend.mem_cmd_tx {
          let mem_cmd = MemCmd {funct, xs1, xs2, rob_id, domain_id};
          if let Err(e) = tx.send(mem_cmd) {
            log_error!("Failed to send MemCmd to MemDomain: {}", e);
          } else {
            log_tpc!("Sent MemCmd to MemDomain: funct={}, rob_id={}, domain_id={}", funct, rob_id, domain_id);
          }
        }
      }
    });
    log_forward!("Inst issue!");
  }
}

impl Sim for Frontend {
  fn forward(&mut self) {
    // self.rocc_cmd();
    // self.enter_rob();
    // self.dispatch_cmd();
    // self.issue_cmd();
  }

  fn backward(&mut self) {
    // 反向出栈并处理：处理队列中的所有事件
    // 每个事件函数会接收 &mut self 作为参数
    // 临时取出队列以避免借用冲突
    let mut queue = std::mem::take(&mut self.event_queue);
    queue.process_all(self);
    self.event_queue = queue;

    // Check if fence instruction is ready (ROB is empty)
    if let Some(fence_rob_id) = self.rob.check_fence_ready() {
      if let Some(ref tx) = self.cmd_response_tx {
        let cmd_response = CmdResponse { result: 0 };
        if let Err(e) = tx.send(cmd_response) {
          log_error!("Failed to send CmdResponse for fence: {}", e);
        } else {
          log_tpc!("Fence instruction completed! fence_rob_id={}, sending cmd_response", fence_rob_id);
        }
      }
    }
  }

  fn module_name(&self) -> &str {
    "Frontend"
  }

  fn print_status(&self) {
    println!("  [Frontend] Module Status:");
    self.decoder.print_status();
    self.rob.print_status();
    self.domain_scheduler.print_status();
    self.rs.print_status();
  }
}
