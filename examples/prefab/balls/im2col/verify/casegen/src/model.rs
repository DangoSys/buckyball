// Reference IM2COL model in the RTL's traversal order (see emu/src/im2col.rs
// and arch/README.md): K tile t outer, window w inner, one write per output
// line, one read per in-bounds tap of a window below window_count.

pub type Row = [u8; 16];

pub fn bank_entries() -> usize {
    let config = std::env::var("BB_VERIFY_CONFIG").expect("BB_VERIFY_CONFIG is required");
    let text = std::fs::read_to_string(config).expect("failed to read BB_VERIFY_CONFIG");
    text.lines()
        .find_map(|line| line.strip_prefix("bank_entries="))
        .expect("bank_entries missing from BB_VERIFY_CONFIG")
        .parse()
        .expect("invalid bank_entries in BB_VERIFY_CONFIG")
}

pub struct Shape {
    pub size: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub start_row: usize,
    pub start_col: usize,
    pub input_base: usize,
    pub window_start: usize,
    pub window_count: usize,
    pub tile_rows: usize,
    pub in_groups: usize,
    pub out_groups: usize,
}

impl Shape {
    pub fn rs2(&self) -> u64 {
        self.kernel as u64
            | (self.stride as u64) << 4
            | (self.padding as u64) << 8
            | (self.start_col as u64) << 12
            | (self.start_row as u64) << 16
            | (self.input_base as u64) << 20
            | (self.window_start as u64) << 26
            | (self.window_count as u64) << 36
            | (self.tile_rows as u64) << 46
    }
}

pub struct Trace {
    // (input group, line) per read, in issue order.
    pub reads: Vec<(usize, usize)>,
    // (output group, line, data) per write, in issue order.
    pub writes: Vec<(usize, usize, Row)>,
}

// `input[g][line]` is input group g; lines below input_base are present too.
pub fn im2col(s: &Shape, input: &[Vec<Row>]) -> Trace {
    let padded = s.size + 2 * s.padding;
    let out_rows = (padded - s.kernel - s.start_row) / s.stride + 1;
    let out_cols = (padded - s.kernel - s.start_col) / s.stride + 1;
    assert!((padded - s.kernel - s.start_row) % s.stride == 0);
    assert!((padded - s.kernel - s.start_col) % s.stride == 0);
    assert!(s.window_start + s.window_count <= out_rows * out_cols);
    assert!(s.input_base + s.size * s.size <= bank_entries());
    let k_tiles = s.kernel * s.kernel * s.in_groups;
    assert!(k_tiles % s.out_groups == 0);
    let group_tiles = k_tiles / s.out_groups;
    let padded_windows = s.window_count.div_ceil(s.tile_rows) * s.tile_rows;
    assert!(padded_windows * group_tiles <= bank_entries());

    let mut trace = Trace { reads: Vec::new(), writes: Vec::new() };
    for tile in 0..k_tiles {
        let g = tile % s.in_groups;
        let kr = tile / s.in_groups / s.kernel;
        let kc = tile / s.in_groups % s.kernel;
        for w in 0..padded_windows {
            let line = ((w / s.tile_rows) * group_tiles + tile % group_tiles) * s.tile_rows
                + w % s.tile_rows;
            let mut data = [0u8; 16];
            if w < s.window_count {
                let window = s.window_start + w;
                let y = (s.start_row + window / out_cols * s.stride + kr) as isize
                    - s.padding as isize;
                let x = (s.start_col + window % out_cols * s.stride + kc) as isize
                    - s.padding as isize;
                if y >= 0 && x >= 0 && (y as usize) < s.size && (x as usize) < s.size {
                    let source = s.input_base + y as usize * s.size + x as usize;
                    trace.reads.push((g, source));
                    data = input[g][source];
                }
            }
            trace.writes.push((tile / group_tiles, line, data));
        }
    }
    trace
}
