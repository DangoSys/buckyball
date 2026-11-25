use super::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old, rs1_update};
use bebop_bemu::{
    ball::blink::{bank_lines, bank_row_bytes},
    ExecContext,
};

fn geometry(rows: u64, groups: u64, n: u64, k: u64, depth: u64) -> bool {
    rows > 0
        && rows <= depth
        && groups > 0
        && n > 0
        && n <= rows * groups * 4
        && k > 0
        && k <= n
        && k <= 2 * depth
}

pub fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    let rows = rs1_iter(xs1);
    let n = xs2 & 0xffff_ffff;
    let k = (xs2 >> 32) & 0xffff;
    let groups = ctx.config(input).cols;
    assert!(
        xs2 >> 48 == 0 && rs1_b1(xs1) == 0 && !rs1_read_old(xs1) && rs1_update(xs1),
        "topk_u32: illegal encoding"
    );
    assert!(
        input != output
            && ctx.config(input).allocated
            && ctx.config(output).allocated
            && ctx.config(output).cols == 1,
        "topk_u32: distinct allocated input and one-group output required"
    );
    assert!(
        geometry(rows, groups, n, k, bank_lines() as u64),
        "topk_u32: invalid key count, K or bank geometry"
    );

    let mut keys = Vec::with_capacity(n as usize);
    for index in 0..n as u32 {
        let group = index as u64 / 4 % groups;
        let row = index as u64 / (4 * groups);
        let lane = index as usize % 4;
        let source = pbank_group(ctx, input, group);
        let offset = row as usize * bank_row_bytes() + lane * 4;
        let key = u32::from_le_bytes(ctx.banks[source][offset..offset + 4].try_into().unwrap());
        keys.push((key, index));
    }
    keys.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let destination = pbank_group(ctx, output, 0);
    for (rank, &(key, index)) in keys.iter().take(k as usize).enumerate() {
        let offset = rank * 8;
        ctx.banks[destination][offset..offset + 4].copy_from_slice(&key.to_le_bytes());
        ctx.banks[destination][offset + 4..offset + 8].copy_from_slice(&index.to_le_bytes());
    }
    if k % 2 != 0 {
        ctx.banks[destination][k as usize * 8..(k as usize + 1) * 8].fill(0);
    }
    0
}

pub fn latency(xs1: u64, xs2: u64, ctx: &ExecContext) -> u64 {
    let rows = rs1_iter(xs1);
    let groups = ctx.config(rs1_b0(xs1)).cols;
    let n = xs2 & 0xffff_ffff;
    let k = (xs2 >> 32) & 0xffff;
    assert!(
        xs2 >> 48 == 0
            && rs1_b1(xs1) == 0
            && !rs1_read_old(xs1)
            && rs1_update(xs1)
            && geometry(rows, groups, n, k, bank_lines() as u64),
        "topk_u32: illegal encoding"
    );
    let input = rs1_b0(xs1);
    let output = rs1_b2(xs1);
    assert!(
        input != output
            && ctx.config(input).allocated
            && ctx.config(output).allocated
            && ctx.config(output).cols == 1,
        "topk_u32: distinct allocated input and one-group output required"
    );
    super::latency::topk_cycles(rows, groups, k)
}

#[cfg(test)]
mod tests {
    use super::geometry;

    #[test]
    fn topk_geometry() {
        assert!(geometry(5, 2, 37, 9, 512));
        assert!(geometry(2, 1, 7, 7, 512));
        assert!(geometry(512, 3, 6144, 1024, 512));
        for (rows, groups, n, k) in [
            (0, 1, 1, 1),
            (513, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 5, 1),
            (1, 1, 4, 0),
            (1, 1, 4, 5),
            (512, 3, 6144, 1025),
        ] {
            assert!(!geometry(rows, groups, n, k, 512));
        }
    }
}
