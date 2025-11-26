/// ReLU computation logic
use super::super::common::VECLANE;

pub struct ReluCompute {
    pub reg_array: Vec<Vec<i8>>,
    pub read_counter: u32,
    pub resp_counter: u32,
    pub write_counter: u32,
}

impl ReluCompute {
    pub fn new() -> Self {
        Self {
            reg_array: vec![vec![0i8; VECLANE]; VECLANE],
            read_counter: 0,
            resp_counter: 0,
            write_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.reg_array = vec![vec![0i8; VECLANE]; VECLANE];
        self.read_counter = 0;
        self.resp_counter = 0;
        self.write_counter = 0;
    }

    pub fn apply_relu(&mut self, row: usize, data: &[i8]) {
        for (col, &val) in data.iter().enumerate() {
            self.reg_array[row][col] = val.max(0);
        }
    }

    pub fn get_row(&self, row: usize) -> Vec<i8> {
        self.reg_array[row].clone()
    }

    pub fn is_read_done(&self) -> bool {
        self.resp_counter >= VECLANE as u32
    }

    pub fn is_write_done(&self) -> bool {
        self.write_counter >= VECLANE as u32
    }

    pub fn should_read(&self) -> bool {
        self.read_counter < VECLANE as u32
    }

    pub fn should_write(&self) -> bool {
        self.write_counter < VECLANE as u32
    }
}
