#[derive(Clone, Debug)]
pub struct NpuConfig {
  pub mem_size: usize,
}

impl NpuConfig {
  pub fn new() -> Self {
    Self { mem_size: 1024 }
  }
}

impl Default for NpuConfig {
  fn default() -> Self {
    Self::new()
  }
}
