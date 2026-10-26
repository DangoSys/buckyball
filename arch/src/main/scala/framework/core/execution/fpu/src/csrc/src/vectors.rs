use super::{Input, Output};

// instruction, integer, a, rhs, c, rounding -> data, flags, integer, illegal
pub const CASES: &[(Input, Output)] = &[
    // S exact arithmetic
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S exact arithmetic
    (
        (
            0x08000053,
            0x0000000000000000,
            0xffffffff40400000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 0, false, false),
    ),
    // S exact arithmetic
    (
        (
            0x10000053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S exact arithmetic
    (
        (
            0x18000053,
            0x0000000000000000,
            0xffffffff40c00000,
            0xffffffff40400000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40000000, 0, false, false),
    ),
    // S sqrt 4
    (
        (
            0x58000053,
            0x0000000000000000,
            0xffffffff40800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40000000, 0, false, false),
    ),
    // S sqrt negative
    (
        (
            0x58000053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 16, false, false),
    ),
    // S divide by zero
    (
        (
            0x18000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7f800000, 8, false, false),
    ),
    // S zero divided by zero
    (
        (
            0x18000053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 16, false, false),
    ),
    // S infinity divided by infinity
    (
        (
            0x18000053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 16, false, false),
    ),
    // S opposite infinities
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffffff800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 16, false, false),
    ),
    // S quiet NaN arithmetic
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffff7fc00011,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 0, false, false),
    ),
    // S signaling NaN arithmetic
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 16, false, false),
    ),
    // S fused sign variants
    (
        (
            0x00000043,
            0x0000000000000000,
            0xffffffff40000000,
            0xffffffff40400000,
            0xffffffff40800000,
            0,
        ),
        (0xffffffff41200000, 0, false, false),
    ),
    // S fused sign variants
    (
        (
            0x00000047,
            0x0000000000000000,
            0xffffffff40000000,
            0xffffffff40400000,
            0xffffffff40800000,
            0,
        ),
        (0xffffffff40000000, 0, false, false),
    ),
    // S fused sign variants
    (
        (
            0x0000004b,
            0x0000000000000000,
            0xffffffff40000000,
            0xffffffff40400000,
            0xffffffff40800000,
            0,
        ),
        (0xffffffffc0000000, 0, false, false),
    ),
    // S fused sign variants
    (
        (
            0x0000004f,
            0x0000000000000000,
            0xffffffff40000000,
            0xffffffff40400000,
            0xffffffff40800000,
            0,
        ),
        (0xffffffffc1200000, 0, false, false),
    ),
    // S fused cancellation retains product precision
    (
        (
            0x00000043,
            0x0000000000000000,
            0xffffffff3f800001,
            0xffffffff3f7ffffe,
            0xffffffffbf800000,
            0,
        ),
        (0xffffffffa8800000, 0, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00000053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff7f7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7f800000, 5, false, false),
    ),
    // S negative overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffffff7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffff800000, 5, false, false),
    ),
    // S positive tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff00000000, 3, false, false),
    ),
    // S negative tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff80000000, 3, false, false),
    ),
    // S cancellation zero sign
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff00000000, 0, false, false),
    ),
    // S float to signed integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // S negative integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffffbfc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00001053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00001053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff7f7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffff7f7fffff, 5, false, false),
    ),
    // S negative overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffffff7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffffff7fffff, 5, false, false),
    ),
    // S positive tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffff00000000, 3, false, false),
    ),
    // S negative tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffff80000000, 3, false, false),
    ),
    // S cancellation zero sign
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff80000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffff00000000, 0, false, false),
    ),
    // S float to signed integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            1,
        ),
        (0x0000000000000001, 1, true, false),
    ),
    // S negative integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffffbfc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            1,
        ),
        (0xffffffffffffffff, 1, true, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00002053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00002053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800001, 1, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffff3f800000, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffffbf800001, 1, false, false),
    ),
    // S positive overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff7f7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffff7f7fffff, 5, false, false),
    ),
    // S negative overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffffff7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffffff800000, 5, false, false),
    ),
    // S positive tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffff00000000, 3, false, false),
    ),
    // S negative tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffff80000001, 3, false, false),
    ),
    // S cancellation zero sign
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff80000000,
            0xffffffff00000000,
            2,
        ),
        (0xffffffff80000000, 0, false, false),
    ),
    // S float to signed integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            2,
        ),
        (0x0000000000000001, 1, true, false),
    ),
    // S negative integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffffbfc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            2,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00003053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800001, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00003053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffff3f800001, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffffbf800000, 1, false, false),
    ),
    // S positive overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff7f7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffff7f800000, 5, false, false),
    ),
    // S negative overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffffff7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffffff7fffff, 5, false, false),
    ),
    // S positive tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffff00000001, 3, false, false),
    ),
    // S negative tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffff80000000, 3, false, false),
    ),
    // S cancellation zero sign
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff80000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffff00000000, 0, false, false),
    ),
    // S float to signed integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            3,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // S negative integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffffbfc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            3,
        ),
        (0xffffffffffffffff, 1, true, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00004053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800001, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00004053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800001, 1, false, false),
    ),
    // S positive half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff33800000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffff3f800001, 1, false, false),
    ),
    // S negative half-ulp rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffffb3800000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffffbf800001, 1, false, false),
    ),
    // S positive overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff7f7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffff7f800000, 5, false, false),
    ),
    // S negative overflow
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffffff7fffff,
            0xffffffff40000000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffffff800000, 5, false, false),
    ),
    // S positive tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffff00000001, 3, false, false),
    ),
    // S negative tiny tie
    (
        (
            0x10007053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff3f000000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffff80000001, 3, false, false),
    ),
    // S cancellation zero sign
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff80000000,
            0xffffffff00000000,
            4,
        ),
        (0xffffffff00000000, 0, false, false),
    ),
    // S float to signed integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            4,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // S negative integer tie
    (
        (
            0xc0207053,
            0x0000000000000000,
            0xffffffffbfc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            4,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // S reserved instruction rounding
    (
        (
            0x00005053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S reserved instruction rounding
    (
        (
            0x00006053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S reserved dynamic rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            5,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S reserved dynamic rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            6,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S reserved dynamic rounding
    (
        (
            0x00007053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            7,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S sign injection
    (
        (
            0x20000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffffc0000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 0, false, false),
    ),
    // S sign injection
    (
        (
            0x20001053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffffc0000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 0, false, false),
    ),
    // S sign injection
    (
        (
            0x20002053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffffc0000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 0, false, false),
    ),
    // S sign injection preserves payload
    (
        (
            0x20000053,
            0x0000000000000000,
            0xffffffff7fc00011,
            0xffffffffc0000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffffc00011, 0, false, false),
    ),
    // S min/max signed zero
    (
        (
            0x28000053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff80000000, 0, false, false),
    ),
    // S min/max finite
    (
        (
            0x28000053,
            0x0000000000000000,
            0xffffffffc0000000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffc0000000, 0, false, false),
    ),
    // S min/max quiet NaN
    (
        (
            0x28000053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 0, false, false),
    ),
    // S min/max signaling NaN
    (
        (
            0x28000053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 16, false, false),
    ),
    // S min/max two NaNs
    (
        (
            0x28000053,
            0x0000000000000000,
            0xffffffff7fc00003,
            0xffffffff7fc00007,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 0, false, false),
    ),
    // S min/max signed zero
    (
        (
            0x28001053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff00000000, 0, false, false),
    ),
    // S min/max finite
    (
        (
            0x28001053,
            0x0000000000000000,
            0xffffffffc0000000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 0, false, false),
    ),
    // S min/max quiet NaN
    (
        (
            0x28001053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 0, false, false),
    ),
    // S min/max signaling NaN
    (
        (
            0x28001053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3f800000, 16, false, false),
    ),
    // S min/max two NaNs
    (
        (
            0x28001053,
            0x0000000000000000,
            0xffffffff7fc00003,
            0xffffffff7fc00007,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 0, false, false),
    ),
    // S comparison ordered
    (
        (
            0xa0000053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // S comparison signed zero
    (
        (
            0xa0000053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // S comparison quiet NaN
    (
        (
            0xa0000053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S comparison signaling NaN
    (
        (
            0xa0000053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S comparison ordered
    (
        (
            0xa0001053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // S comparison signed zero
    (
        (
            0xa0001053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // S comparison quiet NaN
    (
        (
            0xa0001053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S comparison signaling NaN
    (
        (
            0xa0001053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S comparison ordered
    (
        (
            0xa0002053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // S comparison signed zero
    (
        (
            0xa0002053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // S comparison quiet NaN
    (
        (
            0xa0002053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // S comparison signaling NaN
    (
        (
            0xa0002053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S classification 0
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffffff800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // S classification 1
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffffbf800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000002, 0, true, false),
    ),
    // S classification 2
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff80000001,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000004, 0, true, false),
    ),
    // S classification 3
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff80000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000008, 0, true, false),
    ),
    // S classification 4
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000010, 0, true, false),
    ),
    // S classification 5
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff00000001,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000020, 0, true, false),
    ),
    // S classification 6
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000040, 0, true, false),
    ),
    // S classification 7
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000080, 0, true, false),
    ),
    // S classification 8
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff7f800001,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000100, 0, true, false),
    ),
    // S classification 9
    (
        (
            0xe0001053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000200, 0, true, false),
    ),
    // S move float bits to integer
    (
        (
            0xe0000053,
            0x0000000000000000,
            0x0000000080000001,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff80000001, 0, true, false),
    ),
    // S move integer bits to float
    (
        (
            0xf0000053,
            0x0000000080000001,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff80000001, 0, false, false),
    ),
    // S integer conversion type 0
    (
        (
            0xd0000053,
            0x0000000000000003,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S float conversion type 0
    (
        (
            0xc0000053,
            0x0000000000000000,
            0xffffffff40400000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0000053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x000000007fffffff, 16, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0000053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x000000007fffffff, 16, true, false),
    ),
    // S invalid negative conversion
    (
        (
            0xc0000053,
            0x0000000000000000,
            0xffffffffff800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff80000000, 16, true, false),
    ),
    // S integer conversion boundary
    (
        (
            0xd0000053,
            0xffffffffffffffff,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 0, false, false),
    ),
    // S integer conversion type 1
    (
        (
            0xd0100053,
            0x0000000000000003,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S float conversion type 1
    (
        (
            0xc0100053,
            0x0000000000000000,
            0xffffffff40400000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0100053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0100053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // S invalid negative conversion
    (
        (
            0xc0100053,
            0x0000000000000000,
            0xffffffffff800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S integer conversion boundary
    (
        (
            0xd0100053,
            0xffffffffffffffff,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff4f800000, 1, false, false),
    ),
    // S integer conversion type 2
    (
        (
            0xd0200053,
            0x0000000000000003,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S float conversion type 2
    (
        (
            0xc0200053,
            0x0000000000000000,
            0xffffffff40400000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0200053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x7fffffffffffffff, 16, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0200053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x7fffffffffffffff, 16, true, false),
    ),
    // S invalid negative conversion
    (
        (
            0xc0200053,
            0x0000000000000000,
            0xffffffffff800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x8000000000000000, 16, true, false),
    ),
    // S integer conversion boundary
    (
        (
            0xd0200053,
            0xffffffffffffffff,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffbf800000, 0, false, false),
    ),
    // S integer conversion type 3
    (
        (
            0xd0300053,
            0x0000000000000003,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff40400000, 0, false, false),
    ),
    // S float conversion type 3
    (
        (
            0xc0300053,
            0x0000000000000000,
            0xffffffff40400000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0300053,
            0x0000000000000000,
            0xffffffff7f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // S invalid positive conversion
    (
        (
            0xc0300053,
            0x0000000000000000,
            0xffffffff7fc00000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // S invalid negative conversion
    (
        (
            0xc0300053,
            0x0000000000000000,
            0xffffffffff800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // S integer conversion boundary
    (
        (
            0xd0300053,
            0xffffffffffffffff,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff5f800000, 1, false, false),
    ),
    // S cross-format exact
    (
        (
            0x40100053,
            0x0000000000000000,
            0x3ff8000000000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff3fc00000, 0, false, false),
    ),
    // S unboxed single is quiet NaN
    (
        (
            0x00000053,
            0x0000000000000000,
            0x000000003f800000,
            0xffffffff3f800000,
            0xffffffff00000000,
            0,
        ),
        (0xffffffff7fc00000, 0, false, false),
    ),
    // S unboxed single classification
    (
        (
            0xe0001053,
            0x0000000000000000,
            0x000000003f800000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000200, 0, true, false),
    ),
    // S unsupported format
    (
        (
            0x04000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S unsupported format
    (
        (
            0x06000053,
            0x0000000000000000,
            0xffffffff3f800000,
            0xffffffff40000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S unsupported operation
    (
        (
            0xf8000053,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // S unsupported opcode
    (
        (
            0x00000000,
            0x0000000000000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0xffffffff00000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D exact arithmetic
    (
        (
            0x02000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D exact arithmetic
    (
        (
            0x0a000053,
            0x0000000000000000,
            0x4008000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 0, false, false),
    ),
    // D exact arithmetic
    (
        (
            0x12000053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D exact arithmetic
    (
        (
            0x1a000053,
            0x0000000000000000,
            0x4018000000000000,
            0x4008000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4000000000000000, 0, false, false),
    ),
    // D sqrt 4
    (
        (
            0x5a000053,
            0x0000000000000000,
            0x4010000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4000000000000000, 0, false, false),
    ),
    // D sqrt negative
    (
        (
            0x5a000053,
            0x0000000000000000,
            0xbff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 16, false, false),
    ),
    // D divide by zero
    (
        (
            0x1a000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff0000000000000, 8, false, false),
    ),
    // D zero divided by zero
    (
        (
            0x1a000053,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 16, false, false),
    ),
    // D infinity divided by infinity
    (
        (
            0x1a000053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 16, false, false),
    ),
    // D opposite infinities
    (
        (
            0x02000053,
            0x0000000000000000,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 16, false, false),
    ),
    // D quiet NaN arithmetic
    (
        (
            0x02000053,
            0x0000000000000000,
            0x7ff8000000000011,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 0, false, false),
    ),
    // D signaling NaN arithmetic
    (
        (
            0x02000053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 16, false, false),
    ),
    // D fused sign variants
    (
        (
            0x02000043,
            0x0000000000000000,
            0x4000000000000000,
            0x4008000000000000,
            0x4010000000000000,
            0,
        ),
        (0x4024000000000000, 0, false, false),
    ),
    // D fused sign variants
    (
        (
            0x02000047,
            0x0000000000000000,
            0x4000000000000000,
            0x4008000000000000,
            0x4010000000000000,
            0,
        ),
        (0x4000000000000000, 0, false, false),
    ),
    // D fused sign variants
    (
        (
            0x0200004b,
            0x0000000000000000,
            0x4000000000000000,
            0x4008000000000000,
            0x4010000000000000,
            0,
        ),
        (0xc000000000000000, 0, false, false),
    ),
    // D fused sign variants
    (
        (
            0x0200004f,
            0x0000000000000000,
            0x4000000000000000,
            0x4008000000000000,
            0x4010000000000000,
            0,
        ),
        (0xc024000000000000, 0, false, false),
    ),
    // D fused cancellation retains product precision
    (
        (
            0x02000043,
            0x0000000000000000,
            0x3ff0000000000001,
            0x3feffffffffffffe,
            0xbff0000000000000,
            0,
        ),
        (0xb970000000000000, 0, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02000053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0x7fefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7ff0000000000000, 5, false, false),
    ),
    // D negative overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0xffefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xfff0000000000000, 5, false, false),
    ),
    // D positive tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x0000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 3, false, false),
    ),
    // D negative tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x8000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x8000000000000000, 3, false, false),
    ),
    // D cancellation zero sign
    (
        (
            0x02007053,
            0x0000000000000000,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, false),
    ),
    // D float to signed integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // D negative integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0xbff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02001053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02001053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            1,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            1,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0x7fefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            1,
        ),
        (0x7fefffffffffffff, 5, false, false),
    ),
    // D negative overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0xffefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            1,
        ),
        (0xffefffffffffffff, 5, false, false),
    ),
    // D positive tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x0000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            1,
        ),
        (0x0000000000000000, 3, false, false),
    ),
    // D negative tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x8000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            1,
        ),
        (0x8000000000000000, 3, false, false),
    ),
    // D cancellation zero sign
    (
        (
            0x02007053,
            0x0000000000000000,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            1,
        ),
        (0x0000000000000000, 0, false, false),
    ),
    // D float to signed integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            1,
        ),
        (0x0000000000000001, 1, true, false),
    ),
    // D negative integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0xbff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            1,
        ),
        (0xffffffffffffffff, 1, true, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02002053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02002053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000001, 1, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            2,
        ),
        (0x3ff0000000000000, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            2,
        ),
        (0xbff0000000000001, 1, false, false),
    ),
    // D positive overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0x7fefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            2,
        ),
        (0x7fefffffffffffff, 5, false, false),
    ),
    // D negative overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0xffefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            2,
        ),
        (0xfff0000000000000, 5, false, false),
    ),
    // D positive tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x0000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            2,
        ),
        (0x0000000000000000, 3, false, false),
    ),
    // D negative tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x8000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            2,
        ),
        (0x8000000000000001, 3, false, false),
    ),
    // D cancellation zero sign
    (
        (
            0x02007053,
            0x0000000000000000,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            2,
        ),
        (0x8000000000000000, 0, false, false),
    ),
    // D float to signed integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            2,
        ),
        (0x0000000000000001, 1, true, false),
    ),
    // D negative integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0xbff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            2,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02003053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000001, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02003053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            3,
        ),
        (0x3ff0000000000001, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            3,
        ),
        (0xbff0000000000000, 1, false, false),
    ),
    // D positive overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0x7fefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            3,
        ),
        (0x7ff0000000000000, 5, false, false),
    ),
    // D negative overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0xffefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            3,
        ),
        (0xffefffffffffffff, 5, false, false),
    ),
    // D positive tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x0000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            3,
        ),
        (0x0000000000000001, 3, false, false),
    ),
    // D negative tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x8000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            3,
        ),
        (0x8000000000000000, 3, false, false),
    ),
    // D cancellation zero sign
    (
        (
            0x02007053,
            0x0000000000000000,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            3,
        ),
        (0x0000000000000000, 0, false, false),
    ),
    // D float to signed integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            3,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // D negative integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0xbff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            3,
        ),
        (0xffffffffffffffff, 1, true, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02004053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000001, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02004053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000001, 1, false, false),
    ),
    // D positive half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x3ca0000000000000,
            0x0000000000000000,
            4,
        ),
        (0x3ff0000000000001, 1, false, false),
    ),
    // D negative half-ulp rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0xbff0000000000000,
            0xbca0000000000000,
            0x0000000000000000,
            4,
        ),
        (0xbff0000000000001, 1, false, false),
    ),
    // D positive overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0x7fefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            4,
        ),
        (0x7ff0000000000000, 5, false, false),
    ),
    // D negative overflow
    (
        (
            0x12007053,
            0x0000000000000000,
            0xffefffffffffffff,
            0x4000000000000000,
            0x0000000000000000,
            4,
        ),
        (0xfff0000000000000, 5, false, false),
    ),
    // D positive tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x0000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            4,
        ),
        (0x0000000000000001, 3, false, false),
    ),
    // D negative tiny tie
    (
        (
            0x12007053,
            0x0000000000000000,
            0x8000000000000001,
            0x3fe0000000000000,
            0x0000000000000000,
            4,
        ),
        (0x8000000000000001, 3, false, false),
    ),
    // D cancellation zero sign
    (
        (
            0x02007053,
            0x0000000000000000,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            4,
        ),
        (0x0000000000000000, 0, false, false),
    ),
    // D float to signed integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0x3ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            4,
        ),
        (0x0000000000000002, 1, true, false),
    ),
    // D negative integer tie
    (
        (
            0xc2207053,
            0x0000000000000000,
            0xbff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            4,
        ),
        (0xfffffffffffffffe, 1, true, false),
    ),
    // D reserved instruction rounding
    (
        (
            0x02005053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D reserved instruction rounding
    (
        (
            0x02006053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D reserved dynamic rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            5,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D reserved dynamic rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            6,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D reserved dynamic rounding
    (
        (
            0x02007053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            7,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D sign injection
    (
        (
            0x22000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0xc000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 0, false, false),
    ),
    // D sign injection
    (
        (
            0x22001053,
            0x0000000000000000,
            0x3ff0000000000000,
            0xc000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 0, false, false),
    ),
    // D sign injection
    (
        (
            0x22002053,
            0x0000000000000000,
            0x3ff0000000000000,
            0xc000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 0, false, false),
    ),
    // D sign injection preserves payload
    (
        (
            0x22000053,
            0x0000000000000000,
            0x7ff8000000000011,
            0xc000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xfff8000000000011, 0, false, false),
    ),
    // D min/max signed zero
    (
        (
            0x2a000053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x8000000000000000, 0, false, false),
    ),
    // D min/max finite
    (
        (
            0x2a000053,
            0x0000000000000000,
            0xc000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0xc000000000000000, 0, false, false),
    ),
    // D min/max quiet NaN
    (
        (
            0x2a000053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 0, false, false),
    ),
    // D min/max signaling NaN
    (
        (
            0x2a000053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 16, false, false),
    ),
    // D min/max two NaNs
    (
        (
            0x2a000053,
            0x0000000000000000,
            0x7ff8000000000003,
            0x7ff8000000000007,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 0, false, false),
    ),
    // D min/max signed zero
    (
        (
            0x2a001053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, false),
    ),
    // D min/max finite
    (
        (
            0x2a001053,
            0x0000000000000000,
            0xc000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 0, false, false),
    ),
    // D min/max quiet NaN
    (
        (
            0x2a001053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 0, false, false),
    ),
    // D min/max signaling NaN
    (
        (
            0x2a001053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff0000000000000, 16, false, false),
    ),
    // D min/max two NaNs
    (
        (
            0x2a001053,
            0x0000000000000000,
            0x7ff8000000000003,
            0x7ff8000000000007,
            0x0000000000000000,
            0,
        ),
        (0x7ff8000000000000, 0, false, false),
    ),
    // D comparison ordered
    (
        (
            0xa2000053,
            0x0000000000000000,
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // D comparison signed zero
    (
        (
            0xa2000053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // D comparison quiet NaN
    (
        (
            0xa2000053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D comparison signaling NaN
    (
        (
            0xa2000053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D comparison ordered
    (
        (
            0xa2001053,
            0x0000000000000000,
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // D comparison signed zero
    (
        (
            0xa2001053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // D comparison quiet NaN
    (
        (
            0xa2001053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D comparison signaling NaN
    (
        (
            0xa2001053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D comparison ordered
    (
        (
            0xa2002053,
            0x0000000000000000,
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // D comparison signed zero
    (
        (
            0xa2002053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // D comparison quiet NaN
    (
        (
            0xa2002053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, true, false),
    ),
    // D comparison signaling NaN
    (
        (
            0xa2002053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x3ff0000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D classification 0
    (
        (
            0xe2001053,
            0x0000000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000001, 0, true, false),
    ),
    // D classification 1
    (
        (
            0xe2001053,
            0x0000000000000000,
            0xbff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000002, 0, true, false),
    ),
    // D classification 2
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x8000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000004, 0, true, false),
    ),
    // D classification 3
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000008, 0, true, false),
    ),
    // D classification 4
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000010, 0, true, false),
    ),
    // D classification 5
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x0000000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000020, 0, true, false),
    ),
    // D classification 6
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000040, 0, true, false),
    ),
    // D classification 7
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000080, 0, true, false),
    ),
    // D classification 8
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x7ff0000000000001,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000100, 0, true, false),
    ),
    // D classification 9
    (
        (
            0xe2001053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000200, 0, true, false),
    ),
    // D move float bits to integer
    (
        (
            0xe2000053,
            0x0000000000000000,
            0x0123456789abcdef,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0123456789abcdef, 0, true, false),
    ),
    // D move integer bits to float
    (
        (
            0xf2000053,
            0x0123456789abcdef,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0123456789abcdef, 0, false, false),
    ),
    // D integer conversion type 0
    (
        (
            0xd2000053,
            0x0000000000000003,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D float conversion type 0
    (
        (
            0xc2000053,
            0x0000000000000000,
            0x4008000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2000053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x000000007fffffff, 16, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2000053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x000000007fffffff, 16, true, false),
    ),
    // D invalid negative conversion
    (
        (
            0xc2000053,
            0x0000000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xffffffff80000000, 16, true, false),
    ),
    // D integer conversion boundary
    (
        (
            0xd2000053,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 0, false, false),
    ),
    // D integer conversion type 1
    (
        (
            0xd2100053,
            0x0000000000000003,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D float conversion type 1
    (
        (
            0xc2100053,
            0x0000000000000000,
            0x4008000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2100053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2100053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // D invalid negative conversion
    (
        (
            0xc2100053,
            0x0000000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D integer conversion boundary
    (
        (
            0xd2100053,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x41efffffffe00000, 0, false, false),
    ),
    // D integer conversion type 2
    (
        (
            0xd2200053,
            0x0000000000000003,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D float conversion type 2
    (
        (
            0xc2200053,
            0x0000000000000000,
            0x4008000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2200053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7fffffffffffffff, 16, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2200053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x7fffffffffffffff, 16, true, false),
    ),
    // D invalid negative conversion
    (
        (
            0xc2200053,
            0x0000000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x8000000000000000, 16, true, false),
    ),
    // D integer conversion boundary
    (
        (
            0xd2200053,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xbff0000000000000, 0, false, false),
    ),
    // D integer conversion type 3
    (
        (
            0xd2300053,
            0x0000000000000003,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x4008000000000000, 0, false, false),
    ),
    // D float conversion type 3
    (
        (
            0xc2300053,
            0x0000000000000000,
            0x4008000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000003, 0, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2300053,
            0x0000000000000000,
            0x7ff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // D invalid positive conversion
    (
        (
            0xc2300053,
            0x0000000000000000,
            0x7ff8000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0xffffffffffffffff, 16, true, false),
    ),
    // D invalid negative conversion
    (
        (
            0xc2300053,
            0x0000000000000000,
            0xfff0000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 16, true, false),
    ),
    // D integer conversion boundary
    (
        (
            0xd2300053,
            0xffffffffffffffff,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x43f0000000000000, 1, false, false),
    ),
    // D cross-format exact
    (
        (
            0x42000053,
            0x0000000000000000,
            0xffffffff3fc00000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x3ff8000000000000, 0, false, false),
    ),
    // D unsupported format
    (
        (
            0x04000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D unsupported format
    (
        (
            0x06000053,
            0x0000000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D unsupported operation
    (
        (
            0xfa000053,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
    // D unsupported opcode
    (
        (
            0x02000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0x0000000000000000,
            0,
        ),
        (0x0000000000000000, 0, false, true),
    ),
];
