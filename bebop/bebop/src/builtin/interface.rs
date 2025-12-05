/// 调用 Interface 的函数指针的宏
///
/// # 示例
/// ```
/// // 调用静态函数
/// call_interface!(interface, fn(u32, u64, u64), arg1, arg2, arg3);
///
/// // 调用方法（需要传入 self）
/// call_interface!(interface, fn(&mut Self, u32, u64, u64), &mut obj, arg1, arg2, arg3);
/// ```
#[macro_export]
macro_rules! call_interface {
  ($interface:expr, fn($($arg_type:ty),*) $(-> $ret:ty)?, $($arg:expr),*) => {
    if !$interface.function.is_null() {
      unsafe {
        let f: fn($($arg_type),*) $(-> $ret)? = std::mem::transmute($interface.function);
        f($($arg),*)
      }
    }
  };
}

pub struct Interface {
  pub name: String,
  pub latency: u32,
  pub ready: fn() -> bool,
  pub function: *const (),
}

impl Interface {
  pub fn new(name: impl Into<String>, latency: u32) -> Self {
    fn default_ready() -> bool { true }

    Self {
      name: name.into(),
      latency,
      ready: default_ready,
      function: std::ptr::null(),
    }
  }

  pub fn ready(&self) -> bool {
    (self.ready)()
  }

  pub fn set_function(&mut self, f: *const ()) {
    self.function = f;
  }
}
