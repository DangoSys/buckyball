use bebop_bemu::thread::instructions::decode::{pbank, rs1_b0, rs1_b1, rs1_b2, rs1_iter, rs1_read_old};
use bebop_bemu::ball::ExecContext;
pub(super) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let a = rs1_b0(xs1);
    let b = rs1_b1(xs1);
    let c = rs1_b2(xs1);
    let rows = (xs2 & 4095) as usize;
    let k = rs1_iter(xs1) as usize;
    let a_base = ((xs2 >> 12) & 1023) as usize;
    let b_base = ((xs2 >> 22) & 1023) as usize;
    let c_base = ((xs2 >> 32) & 1023) as usize;
    let first = xs2 & (1 << 42) != 0;
    assert_eq!(xs2 >> 43, 0, "redmule: reserved bits must be zero");
    // A continuation reads the old C from wr_read_bank, which rs1[29] provides.
    assert_eq!(
        rs1_read_old(xs1),
        !first,
        "redmule: rs1[29] (read old destination) must be set exactly for a continuation (rs2[42] clear)"
    );
    assert!(
        rows > 0 && k > 0 && k % 8 == 0,
        "redmule: M > 0, K must be a positive multiple of eight"
    );
    assert!(a != b && a != c && b != c, "redmule: banks must differ");
    assert_eq!(
        bebop_bemu::configs::bank_width(),
        128,
        "redmule: bank rows must be 128 bits"
    );
    assert_eq!(bebop_bemu::configs::ball_domain::parameters::<bebop_bemu::configs::ball_domain::parameters::redmule::Params>("redmule", ctx.ball_id.expect("Ball instance")).lanes.expect("lanes") as usize, 8);
    for bank in [a, b, c] {
        assert!(
            ctx.config(bank).allocated && ctx.config(bank).cols == 1,
            "redmule: banks must be allocated with one group"
        );
    }
    let pa = pbank(ctx, a);
    let pb = pbank(ctx, b);
    let pc = pbank(ctx, c);
    assert!(
        a_base + rows * (k / 8) <= ctx.banks[pa].len() / 16
            && b_base + k <= ctx.banks[pb].len() / 16
            && c_base + rows <= ctx.banks[pc].len() / 16,
        "redmule: bank window exceeds capacity"
    );
    let mut sums = vec![0; rows * 8];
    if !first {
        for (index, value) in sums.iter_mut().enumerate() {
            let offset = c_base * 16 + index * 2;
            *value = u16::from_le_bytes(ctx.banks[pc][offset..offset + 2].try_into().unwrap());
        }
    }
    for row in 0..rows {
        for inner in 0..k {
            let offset = a_base * 16 + (row * k + inner) * 2;
            let lhs = u16::from_le_bytes(ctx.banks[pa][offset..offset + 2].try_into().unwrap());
            for col in 0..8 {
                let offset = (b_base + inner) * 16 + col * 2;
                let rhs = u16::from_le_bytes(ctx.banks[pb][offset..offset + 2].try_into().unwrap());
                let index = row * 8 + col;
                sums[index] = super::fp16::fma(lhs, rhs, sums[index]);
            }
        }
    }
    for (index, value) in sums.iter().enumerate() {
        let offset = c_base * 16 + index * 2;
        ctx.banks[pc][offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    0
}
