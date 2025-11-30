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
