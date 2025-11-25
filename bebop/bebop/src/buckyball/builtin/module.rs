/// Base trait for all hardware modules in the NPU simulator
///
/// All modules must implement this trait to participate in cycle-accurate simulation

/// Base trait for hardware modules with cycle-accurate simulation
pub trait Module {
  /// Combinational logic - compute outputs based on current inputs
  fn run(&mut self);

  /// Reset the module to its initial state
  fn reset(&mut self);

  /// Get the module name for debugging/logging
  fn name(&self) -> &str;

  /// Execute one complete clock cycle (run)
  fn tick(&mut self) {
    self.run();
  }
}
