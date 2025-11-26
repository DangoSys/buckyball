/// Matrix computation - simplified matmul
use super::super::common::VECLANE;

pub struct MatrixCompute {
    pub op1_buf: Vec<Vec<i8>>,
    pub op2_buf: Vec<Vec<i8>>,
    pub result_buf: Vec<Vec<i32>>,
    pub read_counter: u32,
    pub resp_counter: u32,
    pub write_counter: u32,
    computed: bool,
}

impl MatrixCompute {
    pub fn new() -> Self {
        Self {
            op1_buf: vec![vec![0i8; VECLANE]; VECLANE],
            op2_buf: vec![vec![0i8; VECLANE]; VECLANE],
            result_buf: vec![vec![0i32; VECLANE]; VECLANE],
            read_counter: 0,
            resp_counter: 0,
            write_counter: 0,
            computed: false,
        }
    }

    pub fn reset(&mut self) {
        self.read_counter = 0;
        self.resp_counter = 0;
        self.write_counter = 0;
        self.computed = false;
    }

    pub fn store_data(&mut self, row: usize, data: &[i8], is_op1: bool) {
        if is_op1 {
            self.op1_buf[row].copy_from_slice(data);
        } else {
            self.op2_buf[row].copy_from_slice(data);
        }
    }

    pub fn compute_matmul(&mut self) {
        if self.computed { return; }
        for i in 0..VECLANE {
            for j in 0..VECLANE {
                let mut sum = 0i32;
                for k in 0..VECLANE {
                    sum += self.op1_buf[i][k] as i32 * self.op2_buf[k][j] as i32;
                }
                self.result_buf[i][j] = sum;
            }
        }
        self.computed = true;
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
