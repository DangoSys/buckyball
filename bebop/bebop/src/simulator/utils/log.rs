/// Logging utilities with colored output

/// Print a log message with blue [Log] prefix
#[macro_export]
macro_rules! log_info {
  ($($arg:tt)*) => {
    println!("\x1b[34m[Log]\x1b[0m {}", format!($($arg)*));
  };
}

#[macro_export]
macro_rules! log_ipc {
  ($($arg:tt)*) => {
    println!("\x1b[32m[IPC]\x1b[0m {}", format!($($arg)*));
  };
}

/// Print an error message with red [Error] prefix
#[macro_export]
macro_rules! log_error {
  ($($arg:tt)*) => {
    eprintln!("\x1b[31m[Error]\x1b[0m {}", format!($($arg)*));
  };
}

/// Print an error message with red [Error] prefix
#[macro_export]
macro_rules! log_event {
  ($($arg:tt)*) => {
    println!("\x1b[33m[Event]\x1b[0m {}", format!($($arg)*));
  };
}

#[macro_export]
macro_rules! log_forward {
  ($($arg:tt)*) => {{
    if $crate::simulator::utils::log_config::is_forward_log_enabled() {
      println!("\x1b[33m[Forward]\x1b[0m {}", format!($($arg)*));
    }
  }};
}

#[macro_export]
macro_rules! log_backward {
  ($($arg:tt)*) => {{
    if $crate::simulator::utils::log_config::is_backward_log_enabled() {
      println!("\x1b[33m[Backward]\x1b[0m {}", format!($($arg)*));
    }
  }};
}

#[macro_export]
macro_rules! log_tpc {
  ($($arg:tt)*) => {{
    println!("\x1b[35m[Tpc]\x1b[0m {}", format!($($arg)*));
  }};
}
