/// Global logging configuration
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flags for controlling log output
static ENABLE_FORWARD_LOG: AtomicBool = AtomicBool::new(true);
static ENABLE_BACKWARD_LOG: AtomicBool = AtomicBool::new(true);

/// Enable or disable forward phase logging
pub fn set_forward_log(enabled: bool) {
    ENABLE_FORWARD_LOG.store(enabled, Ordering::Relaxed);
}

/// Enable or disable backward phase logging
pub fn set_backward_log(enabled: bool) {
    ENABLE_BACKWARD_LOG.store(enabled, Ordering::Relaxed);
}

/// Check if forward logging is enabled
pub fn is_forward_log_enabled() -> bool {
    ENABLE_FORWARD_LOG.load(Ordering::Relaxed)
}

/// Check if backward logging is enabled
pub fn is_backward_log_enabled() -> bool {
    ENABLE_BACKWARD_LOG.load(Ordering::Relaxed)
}

/// Enable all phase logs
pub fn enable_all_logs() {
    set_forward_log(true);
    set_backward_log(true);
}

/// Disable all phase logs
pub fn disable_all_logs() {
    set_forward_log(false);
    set_backward_log(false);
}
