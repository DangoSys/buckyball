/// Im2col computation - simplified
use super::super::common::VECLANE;

pub struct Im2colCompute {
    pub input_buf: Vec<Vec<i8>>,
    pub output_buf: Vec<Vec<i8>>,
    pub read_counter: u32,
    pub resp_counter: u32,
    pub write_counter: u32,
}

impl Im2colCompute {
    pub fn new() -> Self {
        Self {
            input_buf: vec![vec![0i8; VECLANE]; VECLANE],
            output_buf: vec![vec![0i8; VECLANE]; VECLANE],
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

    pub fn store_input(&mut self, row: usize, data: &[i8]) {
        self.input_buf[row].copy_from_slice(data);
    }

    pub fn convert_im2col(&mut self) {
        for i in 0..VECLANE {
            for j in 0..VECLANE {
                self.output_buf[i][j] = self.input_buf[i][j];
            }
        }
    }

    pub fn get_output_row(&self, row: usize) -> Vec<i8> {
        self.output_buf[row].clone()
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
