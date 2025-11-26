/// Ball registration helper
/// Mirrors RTL: examples/toy/balldomain/bbus/busRegister.scala

/// Macro to create a BBus with registered balls
/// Usage:
/// ```
/// let bbus = register_balls![
///   VecBall::new(0),
///   MatrixBall::new(1),
///   ReluBall::new(4),
/// ];
/// ```
#[macro_export]
macro_rules! register_balls {
  ($($ball:expr),* $(,)?) => {
    $crate::builtin::BBus::new(vec![
      $(Box::new($ball) as Box<dyn $crate::builtin::Ball>),*
    ])
  };
}

pub use register_balls;
