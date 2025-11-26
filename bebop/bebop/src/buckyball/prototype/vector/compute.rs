/// Vector computation logic
use super::super::common::VECLANE;

pub struct VecCompute {
  pub op1_buf: Vec<Vec<i8>>,
  pub op2_buf: Vec<Vec<i8>>,
  pub result_buf: Vec<Vec<i32>>,
  pub read_counter: u32,
  pub resp_counter: u32,
  pub write_counter: u32,
}

impl VecCompute {
  pub fn new() -> Self {
    Self {
      op1_buf: vec![vec![0i8; VECLANE]; VECLANE],
      op2_buf: vec![vec![0i8; VECLANE]; VECLANE],
      result_buf: vec![vec![0i32; VECLANE]; VECLANE],
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

  pub fn store_op1(&mut self, row: usize, data: &[i8]) {
    self.op1_buf[row].copy_from_slice(data);
  }

  pub fn store_op2(&mut self, row: usize, data: &[i8]) {
    self.op2_buf[row].copy_from_slice(data);
  }

  pub fn compute_add(&mut self) {
    for row in 0..VECLANE {
      for col in 0..VECLANE {
        self.result_buf[row][col] =
          self.op1_buf[row][col] as i32 + self.op2_buf[row][col] as i32;
      }
    }
  }

  pub fn get_result_row(&self, row: usize) -> Vec<i32> {
    self.result_buf[row].clone()
  }

  pub fn should_read(&self) -> bool {
    self.read_counter < VECLANE as u32 * 2
  }

  pub fn is_read_done(&self) -> bool {
    self.resp_counter >= VECLANE as u32 * 2
  }

  pub fn should_write(&self) -> bool {
    self.write_counter < VECLANE as u32
  }

  pub fn is_write_done(&self) -> bool {
    self.write_counter >= VECLANE as u32
  }
}
