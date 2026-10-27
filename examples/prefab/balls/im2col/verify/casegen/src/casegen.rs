use crate::model::{im2col, Row, Shape};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2colCmd {
    pub iter: u32,
    pub rs2_lo: u32,
    pub rs2_hi: u32,
    pub op1_bank: u32,
    pub wr_bank: u32,
    pub op1_group: u32,
    pub wr_group: u32,
    pub op1_col: u32,
    pub wr_col: u32,
    pub rob_id: u32,
    pub num_src: u32,
    pub num_reads: u32,
    pub num_dst: u32,
}

pub struct Im2colCase {
    pub cmd: Im2colCmd,
    // (input group, line, data) preloaded into the input bank.
    pub src: Vec<(usize, usize, Row)>,
    pub reads: Vec<(usize, usize)>,
    pub dst: Vec<(usize, usize, Row)>,
}

pub fn gen_case(index: u32) -> Im2colCase {
    match index {
        0 => k1_two_groups(),
        1 => k3_nine_output_groups(),
        2 => k3_pad_tile4(),
        3 => asym_pad(),
        4 => partial_window_range(),
        5 => k5_tile_rows_1(),
        6 => k7_stride2(),
        7 => window_input_base_two_groups(),
        _ => panic!("im2col: unsupported directed case {index}"),
    }
}

fn finish(
    shape: Shape,
    input: Vec<Vec<Row>>,
    banks: (u32, u32),
    groups: (u32, u32),
    rob_id: u32,
) -> Im2colCase {
    let trace = im2col(&shape, &input);
    let rs2 = shape.rs2();
    let mut src = Vec::new();
    for (g, rows) in input.iter().enumerate() {
        for (line, row) in rows.iter().enumerate() {
            src.push((g, line, *row));
        }
    }
    Im2colCase {
        cmd: Im2colCmd {
            iter: shape.size as u32,
            rs2_lo: rs2 as u32,
            rs2_hi: (rs2 >> 32) as u32,
            op1_bank: banks.0,
            wr_bank: banks.1,
            op1_group: groups.0,
            wr_group: groups.1,
            op1_col: shape.in_groups as u32,
            wr_col: shape.out_groups as u32,
            rob_id,
            num_src: src.len() as u32,
            num_reads: trace.reads.len() as u32,
            num_dst: trace.writes.len() as u32,
        },
        src,
        reads: trace.reads,
        dst: trace.writes,
    }
}

fn row(seed: usize) -> Row {
    let mut r = [0u8; 16];
    for (lane, byte) in r.iter_mut().enumerate() {
        *byte = (seed * 16 + lane) as u8;
    }
    r
}

// 1x1 kernel on a 2x2 input, 32 channels: output group g copies input group g.
fn k1_two_groups() -> Im2colCase {
    let input = vec![
        (0..4).map(|p| row(p * 2 + 1)).collect(),
        (0..4).map(|p| row(p * 2 + 40)).collect(),
    ];
    let shape = Shape {
        size: 2, kernel: 1, stride: 1, padding: 0, start_row: 0, start_col: 0,
        input_base: 0, window_start: 0, window_count: 4, tile_rows: 4,
        in_groups: 2, out_groups: 2,
    };
    finish(shape, input, (0, 1), (0, 0), 1)
}

// 3x3 kernel on a 4x4 input: 2x2 windows, nine output groups (one tap each),
// lines 4..7 of each 8-row M tile are zero.
fn k3_nine_output_groups() -> Im2colCase {
    let input = vec![(0..16).map(|p| row(p * 3 + 2)).collect()];
    let shape = Shape {
        size: 4, kernel: 3, stride: 1, padding: 0, start_row: 0, start_col: 0,
        input_base: 0, window_start: 0, window_count: 4, tile_rows: 8,
        in_groups: 1, out_groups: 9,
    };
    finish(shape, input, (2, 5), (1, 3), 2)
}

// 3x3 kernel, padding 1, 2x2 input: nine taps in one group, 4-row M tile.
fn k3_pad_tile4() -> Im2colCase {
    let input = vec![(0..4).map(|p| row(p + 7)).collect()];
    let shape = Shape {
        size: 2, kernel: 3, stride: 1, padding: 1, start_row: 0, start_col: 0,
        input_base: 0, window_start: 0, window_count: 4, tile_rows: 4,
        in_groups: 1, out_groups: 1,
    };
    finish(shape, input, (3, 1), (0, 2), 3)
}

// Bottom/right padding only: 2x2 kernel, padding 1, start 1/1, 3x3 input.
fn asym_pad() -> Im2colCase {
    let input = vec![(0..9).map(|p| row(p * 5 + 1)).collect()];
    let shape = Shape {
        size: 3, kernel: 2, stride: 1, padding: 1, start_row: 1, start_col: 1,
        input_base: 0, window_start: 0, window_count: 9, tile_rows: 16,
        in_groups: 1, out_groups: 4,
    };
    finish(shape, input, (0, 4), (0, 0), 4)
}

// Windows 5..7 of a 1x1 im2col on a 4x4 input at input_base 2; lines 3..7 zero.
fn partial_window_range() -> Im2colCase {
    let input = vec![(0..18).map(|p| row(p + 100)).collect()];
    let shape = Shape {
        size: 4, kernel: 1, stride: 1, padding: 0, start_row: 0, start_col: 0,
        input_base: 2, window_start: 5, window_count: 3, tile_rows: 8,
        in_groups: 1, out_groups: 1,
    };
    finish(shape, input, (1, 0), (2, 1), 5)
}

// 5x5 kernel on a 5x5 input with tile_rows 1: output group kr is image row kr.
fn k5_tile_rows_1() -> Im2colCase {
    let input = vec![(0..25).map(|p| row(p * 7 + 3)).collect()];
    let shape = Shape {
        size: 5, kernel: 5, stride: 1, padding: 0, start_row: 0, start_col: 0,
        input_base: 0, window_start: 0, window_count: 1, tile_rows: 1,
        in_groups: 1, out_groups: 5,
    };
    finish(shape, input, (0, 1), (0, 0), 6)
}

// 7x7 stride-2 kernel on a 9x9 input: 2x2 windows, one kernel row per group.
fn k7_stride2() -> Im2colCase {
    let input = vec![(0..81).map(|p| row(p * 11 + 5)).collect()];
    let shape = Shape {
        size: 9, kernel: 7, stride: 2, padding: 0, start_row: 0, start_col: 0,
        input_base: 0, window_start: 0, window_count: 4, tile_rows: 4,
        in_groups: 1, out_groups: 7,
    };
    finish(shape, input, (6, 2), (1, 0), 7)
}

// 2x2 kernel on a 3x3 input with 32 channels at input_base 3, windows 1..2:
// eight output groups, one (tap, channel group) tile each.
fn window_input_base_two_groups() -> Im2colCase {
    let input = vec![
        (0..12).map(|p| row(p * 3 + 9)).collect(),
        (0..12).map(|p| row(p * 3 + 60)).collect(),
    ];
    let shape = Shape {
        size: 3, kernel: 2, stride: 1, padding: 0, start_row: 0, start_col: 0,
        input_base: 3, window_start: 1, window_count: 2, tile_rows: 2,
        in_groups: 2, out_groups: 8,
    };
    finish(shape, input, (4, 7), (0, 2), 8)
}
