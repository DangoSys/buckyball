use crate::inst::decode::{pbank_group, rs1_b0, rs1_b1, rs1_b2, rs1_iter};
use crate::inst::instruction::ExecContext;

pub(crate) fn execute(xs1: u64, xs2: u64, ctx: &mut ExecContext) -> u64 {
    let a = rs1_b0(xs1);
    let b = rs1_b1(xs1);
    let c = rs1_b2(xs1);
    let rows = rs1_iter(xs1) as usize;
    let group = xs2 & 31;
    let first = xs2 & 32 != 0;
    assert_eq!(xs2 & !63, 0, "f32add: reserved rs2 bits must be zero");
    assert!(a != b && a != c && b != c, "f32add: banks must differ");
    let ac = *ctx.config(a);
    let bc = *ctx.config(b);
    let cc = *ctx.config(c);
    assert!(
        ac.allocated && bc.allocated && cc.allocated,
        "f32add: banks must be allocated"
    );
    assert!(
        group < ac.cols && bc.cols == 1 && cc.cols == 1,
        "f32add: source group must exist and accumulators must have one group"
    );
    let pa = pbank_group(ctx, a, group);
    let pb = pbank_group(ctx, b, 0);
    let pc = pbank_group(ctx, c, 0);
    assert!(
        rows > 0
            && rows <= ctx.banks[pa].len() / 16
            && rows <= ctx.banks[pb].len() / 16
            && rows <= ctx.banks[pc].len() / 16,
        "f32add: row span exceeds a bank"
    );
    for offset in (0..rows * 16).step_by(4) {
        let value = f32::from_le_bytes(ctx.banks[pa][offset..offset + 4].try_into().unwrap());
        let accumulator = if first {
            0.0
        } else {
            f32::from_le_bytes(ctx.banks[pb][offset..offset + 4].try_into().unwrap())
        };
        let sum = accumulator + value;
        let bits = if sum.is_nan() {
            0x7fc00000
        } else {
            sum.to_bits()
        };
        ctx.banks[pc][offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
    }
    0
}
