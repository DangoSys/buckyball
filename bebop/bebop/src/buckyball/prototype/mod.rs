/// Prototype accelerators - each ball is independent with its own ISA
pub mod common;
pub mod vector;
pub mod matrix;
pub mod im2col;
pub mod transpose;
pub mod relu;

pub use vector::VecBall;
pub use matrix::MatrixBall;
pub use im2col::Im2colBall;
pub use transpose::TransposeBall;
pub use relu::ReluBall;
