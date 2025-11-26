/// Ball framework - standard interface for Ball devices
/// Mirrors RTL: framework/blink/ and framework/bbus/
mod blink;
mod traits;
mod bbus;
mod register;

pub use blink::*;
pub use traits::{Ball, Blink};
pub use bbus::BBus;
pub use register::register_balls;
