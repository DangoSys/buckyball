/// GlobalRS - 全局保留站
/// 管理指令的发射和完成

pub type RobId = usize;

/// 保留站条目
#[derive(Clone, Default)]
pub struct RobEntry {
  pub valid: bool,
  pub funct: u64,
  pub xs1: u64,
  pub xs2: u64,
}

/// 全局保留站
pub struct GlobalRS {
  name: String,
  rob: Vec<RobEntry>,
  head: usize,
  tail: usize,
}

impl GlobalRS {
  pub fn new(name: impl Into<String>, size: usize) -> Self {
    Self {
      name: name.into(),
      rob: vec![RobEntry::default(); size],
      head: 0,
      tail: 0,
    }
  }

  /// 分配 ROB 条目
  pub fn allocate(&mut self, funct: u64, xs1: u64, xs2: u64) -> Option<RobId> {
    let next_tail = (self.tail + 1) % self.rob.len();
    if next_tail == self.head {
      return None; // ROB full
    }

    let rob_id = self.tail;
    self.rob[rob_id] = RobEntry { valid: true, funct, xs1, xs2 };
    self.tail = next_tail;
    Some(rob_id)
  }

  /// 提交完成的指令
  pub fn commit(&mut self, rob_id: RobId) {
    if rob_id < self.rob.len() && self.rob[rob_id].valid {
      self.rob[rob_id].valid = false;
      // 更新 head
      while self.head != self.tail && !self.rob[self.head].valid {
        self.head = (self.head + 1) % self.rob.len();
      }
    }
  }

  pub fn is_full(&self) -> bool {
    (self.tail + 1) % self.rob.len() == self.head
  }

  #[allow(dead_code)]
  pub fn is_empty(&self) -> bool {
    self.head == self.tail
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &str {
    &self.name
  }
}
