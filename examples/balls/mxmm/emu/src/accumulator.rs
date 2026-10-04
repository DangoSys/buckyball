use std::{cell::RefCell, collections::HashMap};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Mxfp8,
    Mxfp8Window,
    Fma32,
    F32,
}

pub(crate) struct Chain {
    pub(crate) format: Format,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) bank: u64,
    pub(crate) base: usize,
    pub(crate) values: Vec<f32>,
}

thread_local! {
    pub(crate) static CHAINS: RefCell<HashMap<usize, Chain>> = RefCell::new(HashMap::new());
}
