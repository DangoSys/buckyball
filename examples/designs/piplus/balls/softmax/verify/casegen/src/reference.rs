//! SOFTMAX reference, copied from emu/src/softmax.rs: every FP32 operation in
//! the order workloads/isa/softmax.h specifies, so the RTL result is bit-exact.

pub const LOG2E: f32 = 1.442_695_040_888_963_4;
/// P(f) ~ 2^f on [0, 1): c1 .. c6.
const COEFFICIENTS: [f32; 6] = [
    0.693_147_180_559_945_3,
    0.240_226_506_959_100_7,
    0.055_504_108_664_821_6,
    0.009_618_129_107_628_5,
    0.001_333_355_814_642_8,
    0.000_154_035_303_933_8,
];

/// 2^u of u <= 0, zero below -126.
fn exp2(u: f32) -> f32 {
    if u < -126.0 {
        return 0.0;
    }
    let n = u.floor();
    let f = u - n;
    let mut p = COEFFICIENTS[5];
    for c in COEFFICIENTS[..5].iter().rev() {
        p = p * f + c;
    }
    p = p * f + 1.0;
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

pub fn maximum(x: &[f32]) -> f32 {
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = maximum(x);
    let e: Vec<f32> = x.iter().map(|&v| exp2((v - m) * LOG2E)).collect();
    let mut lanes = [0.0f32; 4];
    for (i, &v) in e.iter().enumerate() {
        lanes[i % 4] += v;
    }
    let s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    let r = 1.0 / s;
    e.iter().map(|&v| v * r).collect()
}

/// Bank images of one command: `input` holds `rows` rows of w FP32 values from
/// bank row `in_base`, the result goes to a copy of `output` from `out_base`.
/// Returns the semantic events the rows hit, see `Case::fields`.
pub fn execute(
    input: &[u8],
    output: &mut [u8],
    rows: usize,
    w: usize,
    in_base: usize,
    out_base: usize,
    scale_log2: i32,
) -> [u32; 7] {
    let lines = w.div_ceil(4);
    let scale = f32::from_bits(((scale_log2 + 127) as u32) << 23);
    let mut events = [0u32; 7];
    output[out_base * 16..(out_base + rows * lines) * 16].fill(0);
    for row in 0..rows {
        let at = (in_base + row * lines) * 16;
        let x: Vec<f32> = input[at..at + w * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) * scale)
            .collect();
        assert!(x.iter().all(|v| v.is_finite() || *v == f32::NEG_INFINITY));
        assert!(x.iter().any(|v| v.is_finite()));
        let m = maximum(&x);
        let y = softmax(&x);
        events[3] += u32::from(x.iter().filter(|&&v| v == m).count() > 1);
        for (i, (&v, &r)) in x.iter().zip(y.iter()).enumerate() {
            let u = (v - m) * LOG2E;
            events[0] += u32::from(v.is_finite() && u < -126.0);
            events[1] += u32::from(v == f32::NEG_INFINITY);
            events[2] += u32::from(v.is_finite() && v - m == f32::NEG_INFINITY);
            events[4] += u32::from(v.is_subnormal());
            events[5] += u32::from(r.is_subnormal());
            events[6] += u32::from(v.is_finite() && (-127.0..=-125.0).contains(&u));
            let at = ((out_base + row * lines) * 4 + i) * 4;
            output[at..at + 4].copy_from_slice(&r.to_le_bytes());
        }
    }
    events
}
