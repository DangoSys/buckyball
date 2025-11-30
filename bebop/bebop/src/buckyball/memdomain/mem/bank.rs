/// Memory Bank - simple scratchpad storage
pub struct Bank {
  name: String,
  spad: Vec<u32>,
}

impl Bank {
  pub fn new(name: impl Into<String>, size: usize) -> Self {
    Self {
      name: name.into(),
      spad: vec![0; size],
    }
  }

  /// Direct write data
  pub fn init_write(&mut self, addr: usize, data: u32) {
    if addr < self.spad.len() {
      self.spad[addr] = data;
    }
  }

  /// Direct read data
  pub fn read_data(&self, addr: usize) -> u32 {
    if addr < self.spad.len() {
      self.spad[addr]
    } else {
      0
    }
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &str {
    &self.name
  }
}
