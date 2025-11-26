/// Transpose computation
use super::super::common::VECLANE;

pub struct TransposeCompute {
    pub matrix_buf: Vec<Vec<i8>>,
    pub read_counter: u32,
    pub resp_counter: u32,
    pub write_counter: u32,
}

impl TransposeCompute {
    pub fn new() -> Self {
        Self {
            matrix_buf: vec![vec![0i8; VECLANE]; VECLANE],
            read_counter: 0,
            resp_counter: 0,
            write_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.read_counter = 0;
        self.resp_counter = 0;
        self.write_counter = 0;
    }

    pub fn store_row(&mut self, row: usize, data: &[i8]) {
        self.matrix_buf[row].copy_from_slice(data);
    }

    pub fn get_transposed_row(&self, row: usize) -> Vec<i8> {
        (0..VECLANE).map(|col| self.matrix_buf[col][row]).collect()
    }

    pub fn should_read(&self) -> bool {
        self.read_counter < VECLANE as u32
    }

    pub fn is_read_done(&self) -> bool {
        self.resp_counter >= VECLANE as u32
    }

    pub fn should_write(&self) -> bool {
        self.write_counter < VECLANE as u32
    }

    pub fn is_write_done(&self) -> bool {
        self.write_counter >= VECLANE as u32
    }
}
