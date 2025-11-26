/// Port and signal types for module interconnection

/// A wire/signal that carries data between modules
/// All signal lines automatically include valid flag
#[derive(Clone)]
pub struct Wire<T: Clone> {
  pub value: T,
  pub valid: bool,
}

impl<T: Clone> Wire<T> {
  pub fn new(value: T) -> Self {
    Self { value, valid: false }
  }

  pub fn set(&mut self, value: T) {
    self.value = value;
    self.valid = true;
  }

  pub fn clear(&mut self) {
    self.valid = false;
  }
}

impl<T: Clone + Default> Default for Wire<T> {
  fn default() -> Self {
    Self {
      value: T::default(),
      valid: false,
    }
  }
}
