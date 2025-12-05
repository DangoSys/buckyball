pub trait Sim {
  fn module_name(&self) -> &str;
  fn forward(&mut self);
  fn backward(&mut self);

  /// Optional method to print module status
  /// Override this to print module-specific state
  fn print_status(&self) {
    // Default: do nothing
  }
}
