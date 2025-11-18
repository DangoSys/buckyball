/// NPU configuration parameters

/// NPU configuration
#[derive(Clone, Debug)]
pub struct NpuConfig {
  /// Scratchpad memory size (number of u32)
  pub mem_size: usize,
  // More parameters can be added in the future, for example:
  // pub num_pe: usize,           // Number of PEs
  // pub cache_size: usize,        // Cache size
  // pub max_batch_size: usize,    // Maximum batch size
  // pub clock_freq_mhz: u32,      // Clock frequency
}

impl NpuConfig {
  /// Create default configuration
  pub fn new() -> Self {
    Self {
      mem_size: 1024, // Default 1024 u32 = 4KB
    }
  }

  /// Custom configuration
  pub fn with_mem_size(mem_size: usize) -> Self {
    Self { mem_size }
  }
}

impl Default for NpuConfig {
  fn default() -> Self {
    Self::new()
  }
}
