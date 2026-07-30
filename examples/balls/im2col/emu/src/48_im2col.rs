use super::super::bank::BANK_NUM;
use super::decode::{pbank, pbank_group, rs1_b0, rs1_b2, rs1_iter};
use super::instruction::{ExecContext, Instruction};

pub struct Im2col;

impl Instruction for Im2col {
    const FUNCT: u32 = 48;

    fn exec(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
        let op1 = rs1_b0(xs1);
        let wr = rs1_b2(xs1);

        if op1 >= bank_num() as u64 || wr >= bank_num() as u64 {
            panic!("im2col: invalid bank_id");
        }
        if !ctx.cfgs[op1 as usize].allocated || !ctx.cfgs[wr as usize].allocated {
            panic!("im2col: bank not allocated");
        }
        if op1 == wr {
            panic!("im2col: op1 and wr must differ");
        }

        let iter = rs1_iter(xs1) as usize;
        let ksize = (xs2 & 0xFF) as usize;
        let stride = ((xs2 >> 8) & 0xFF) as usize;
        let padding = ((xs2 >> 16) & 0xFF) as usize;

        if iter == 0 || ksize == 0 || stride == 0 {
            panic!("im2col: invalid shape (zero dim)");
        }
        let padded_size = iter + 2 * padding;
        if padded_size < ksize {
            panic!("im2col: kernel larger than padded input");
        }

        let output_dim = (padded_size - ksize) / stride + 1;

        const TILE: usize = 16;
        let po = pbank(ctx.bank_map, op1);
        let windows = output_dim * output_dim;
        let kernel_elems = ksize * ksize;
        let m_tiles = windows.div_ceil(TILE);
        let k_tiles = kernel_elems.div_ceil(TILE);
        let mut output = vec![0u8; m_tiles * k_tiles * TILE * TILE];

        let srcb = &ctx.banks[po];
        let mut window = 0usize;
        for output_row in 0..output_dim {
            for output_col in 0..output_dim {
                for kr in 0..ksize {
                    for kc in 0..ksize {
                        let input_row = (output_row * stride + kr) as isize - padding as isize;
                        let input_col = (output_col * stride + kc) as isize - padding as isize;
                        let kernel = kr * ksize + kc;
                        let m_tile = window / TILE;
                        let m_row = window % TILE;
                        let k_tile = kernel / TILE;
                        let lane = kernel % TILE;
                        let bank_row = (m_tile * k_tiles + k_tile) * TILE + m_row;
                        let out = bank_row * TILE + lane;
                        if out >= output.len() {
                            panic!("im2col: output range out={out}");
                        }
                        if input_row >= 0
                            && input_row < iter as isize
                            && input_col >= 0
                            && input_col < iter as isize
                        {
                            let src = input_row as usize * iter + input_col as usize;
                            if src >= srcb.len() {
                                panic!("im2col: input range src={src}");
                            }
                            output[out] = srcb[src];
                        }
                    }
                }
                window += 1;
            }
        }

        let groups = ctx.cfgs[wr as usize].cols as usize;
        let rows_per_bank = ctx.banks[pbank_group(ctx.bank_map, wr, 0)].len() / TILE;
        for (row, data) in output.chunks_exact(TILE).enumerate() {
            let group = row / rows_per_bank;
            let local_row = row % rows_per_bank;
            if group >= groups {
                panic!("im2col: output requires group {group}, allocated groups={groups}");
            }
            let pw = pbank_group(ctx.bank_map, wr, group as u64);
            let off = local_row * TILE;
            ctx.banks[pw][off..off + TILE].copy_from_slice(data);
        }
        0
    }

    fn latency(_xs1: u64, xs2: u64) -> u64 {
        let iter = rs1_iter(_xs1);
        let ksize = xs2 & 0xFF;
        let stride = (xs2 >> 8) & 0xFF;
        let padding = (xs2 >> 16) & 0xFF;

        if iter == 0 || ksize == 0 || stride == 0 {
            return 16;
        }
        let padded_size = iter + 2 * padding;
        if padded_size < ksize {
            return 16;
        }

        let output_dim = (padded_size - ksize) / stride + 1;
        output_dim
            .saturating_mul(output_dim)
            .saturating_mul(ksize)
            .saturating_mul(ksize)
            .max(16)
    }
}
