//! LAYERNORM reference, operation for operation as workloads/isa/layernorm.h
//! and the BEMU model (emu/src/layernorm.rs) define it: FP32, round to
//! nearest even, so the RTL output must match bit for bit.

/// Lane sums in increasing element order, then (s0 + s1) + (s2 + s3).
fn sum(values: impl Iterator<Item = f32>) -> f32 {
    let mut lanes = [0.0f32; 4];
    for (i, v) in values.enumerate() {
        lanes[i % 4] += v;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// With rms (`beta` None): the mean is 0 and nothing is added.
fn layernorm(x: &[f32], gamma: &[f32], beta: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let inverse = 1.0 / x.len() as f32;
    let mean = if beta.is_some() {
        sum(x.iter().copied()) * inverse
    } else {
        0.0
    };
    let d: Vec<f32> = x.iter().map(|&v| v - mean).collect();
    let r = 1.0 / (sum(d.iter().map(|&v| v * v)) * inverse + eps).sqrt();
    d.iter()
        .enumerate()
        .map(|(i, &v)| match beta {
            Some(beta) => v * r * gamma[i] + beta[i],
            None => v * r * gamma[i],
        })
        .collect()
}

fn floats(words: &[u32]) -> Vec<f32> {
    words.iter().map(|&b| f32::from_bits(b)).collect()
}

/// The output bank image: `guard` holds the rows outside the window; a row's
/// lanes after w are zero.
#[allow(clippy::too_many_arguments)]
pub fn expected(
    input: &[u32],
    params: &[u32],
    guard: &[u32],
    rows: usize,
    w: usize,
    in_base: usize,
    out_base: usize,
    param_base: usize,
    rms: bool,
) -> Vec<u32> {
    let lines = w.div_ceil(4);
    let gamma = floats(&params[param_base * 4..param_base * 4 + w]);
    let beta = if rms {
        Vec::new()
    } else {
        floats(&params[(param_base + lines) * 4..(param_base + lines) * 4 + w])
    };
    let eps_at = param_base + if rms { lines } else { 2 * lines };
    let eps = f32::from_bits(params[eps_at * 4]);
    let mut output = guard.to_vec();
    for row in 0..rows {
        let at = (in_base + row * lines) * 4;
        let x = floats(&input[at..at + w]);
        let y = layernorm(&x, &gamma, (!rms).then_some(&beta[..]), eps);
        assert!(
            x.iter()
                .chain(&gamma)
                .chain(if rms { &[][..] } else { &beta[..] })
                .chain(&y)
                .all(|v| v.is_finite())
                && eps.is_finite(),
            "layernorm case: a value not finite (row {row})"
        );
        let to = (out_base + row * lines) * 4;
        output[to..to + lines * 4].fill(0);
        for (i, v) in y.iter().enumerate() {
            output[to + i] = v.to_bits();
        }
    }
    output
}
