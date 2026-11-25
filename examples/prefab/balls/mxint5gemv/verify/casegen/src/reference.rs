use std::sync::Mutex;

const KS: [usize; 10] = [32, 64, 512, 1024, 2944, 512, 32, 1024, 32, 64];
struct Case {
    words: Vec<Vec<u128>>,
    k: usize,
    accumulate: bool,
    finalize: bool,
}
static CURRENT: Mutex<Option<Case>> = Mutex::new(None);

fn rne(value: i128, shift: i32) -> i128 {
    if shift <= 0 {
        return value << -shift;
    }
    if shift >= 127 {
        return 0;
    }
    let a = value.abs();
    let q = a >> shift;
    let rem = a - (q << shift);
    let half = 1i128 << (shift - 1);
    let result = q + i128::from(rem > half || rem == half && q & 1 != 0);
    if value < 0 {
        -result
    } else {
        result
    }
}

fn a(index: usize, k: usize) -> i32 {
    if index == 9 && k % 32 == 0 {
        -128
    } else if index == 7 {
        0
    } else {
        ((k * 11 + index * 3) % 255) as i32 - 127
    }
}
fn b(index: usize, column: usize, k: usize) -> i32 {
    if index == 8 {
        0
    } else {
        ((k * 7 + column * 13 + index) % 31) as i32 - 15
    }
}
fn factor(index: usize, column: usize, block: usize) -> i32 {
    [1, 17, 137][(index + column + block) % 3]
}
fn ea(index: usize, block: usize) -> i32 {
    127 + ((index + block * 3) % 15) as i32 - 7
}
fn eb(index: usize, column: usize, block: usize) -> i32 {
    127 + ((index + column + block * 5) % 17) as i32 - 8
}

// Independent wide-integer reference. No Ball/BEMU implementation is linked.
fn compute(index: usize, k: usize, values: &mut [i128; 32], scale: &mut i32, valid: &mut bool) {
    for block in 0..k / 32 {
        let mut products = [0i128; 32];
        let mut scales = [0i32; 32];
        for col in 0..32 {
            products[col] = (0..32)
                .map(|i| {
                    i128::from(a(index, block * 32 + i)) * i128::from(b(index, col, block * 32 + i))
                })
                .sum::<i128>()
                * i128::from(factor(index, col, block));
            scales[col] = ea(index, block) + eb(index, col, block) - 258;
        }
        let incoming = (0..32)
            .filter(|&col| products[col] != 0)
            .map(|col| scales[col])
            .max();
        if let Some(incoming) = incoming {
            let next = if *valid {
                incoming.max(*scale)
            } else {
                incoming
            };
            if *valid && next > *scale {
                for value in values.iter_mut() {
                    *value = rne(*value, next - *scale);
                }
            }
            for col in 0..32 {
                values[col] += rne(products[col], next - scales[col] - 28);
            }
            *valid = values.iter().any(|&v| v != 0);
            *scale = if *valid { next } else { 0 };
        }
    }
}
fn state(values: &[i128; 32], count: usize, scale: i32, valid: bool) -> Vec<u8> {
    let mut bytes = vec![0u8; 272];
    bytes[0..2].copy_from_slice(&(count as u16).to_le_bytes());
    bytes[2..4].copy_from_slice(&(scale as i16).to_le_bytes());
    bytes[4] = valid as u8;
    bytes[5] = 5;
    bytes[6] = 28;
    for col in 0..32 {
        bytes[16 + col * 8..24 + col * 8]
            .copy_from_slice(&i64::try_from(values[col]).unwrap().to_le_bytes());
    }
    bytes
}
fn pack(bytes: &[u8]) -> Vec<u128> {
    bytes
        .chunks(16)
        .map(|chunk| u128::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
fn set_byte(words: &mut [u128], at: usize, value: u8) {
    let shift = (at % 16) * 8;
    words[at / 16] = words[at / 16] & !(255u128 << shift) | u128::from(value) << shift;
}
fn make(index: usize) -> Case {
    let k = KS[index];
    let accumulate = index >= 5;
    let finalize = index % 2 != 0;
    let code_rows = (k * 5 / 8 + 15) / 16;
    let meta_rows = (k + 511) / 512;
    let stride = code_rows + 2 * meta_rows;
    assert!(3 + stride * 8 <= 1024);
    let guard = u128::from_le_bytes([0x5a; 16]);
    let mut words = vec![vec![guard; 1024]; 7];
    for i in 0..k {
        set_byte(&mut words[0], 2 * 16 + i, a(index, i) as u8);
    }
    for block in 0..k / 32 {
        set_byte(&mut words[0], 500 * 16 + block, ea(index, block) as u8);
    }
    for col in 0..32 {
        let group = col / 8 + 1;
        let base = (3 + col % 8 * stride) * 16;
        // Zero the code padding, then pack five-bit signed codes.
        for i in 0..code_rows * 16 {
            set_byte(&mut words[group], base + i, 0);
        }
        for i in 0..k {
            let bit = i * 5;
            let code = (b(index, col, i) & 31) as u8;
            let at = base + bit / 8;
            let old = ((words[group][at / 16] >> (at % 16 * 8)) & 255) as u8;
            set_byte(&mut words[group], at, old | code << (bit % 8));
            if bit % 8 > 3 {
                let at = at + 1;
                let old = ((words[group][at / 16] >> (at % 16 * 8)) & 255) as u8;
                set_byte(&mut words[group], at, old | code >> (8 - bit % 8));
            }
        }
        for block in 0..k / 32 {
            set_byte(
                &mut words[group],
                base + code_rows * 16 + block,
                factor(index, col, block) as u8,
            );
            set_byte(
                &mut words[group],
                base + (code_rows + meta_rows) * 16 + block,
                eb(index, col, block) as u8,
            );
        }
    }
    let mut values = [0i128; 32];
    let mut scale = 0;
    let mut valid = false;
    if accumulate {
        compute(index, 32, &mut values, &mut scale, &mut valid);
        words[5][5..22].copy_from_slice(&pack(&state(&values, 1, scale, valid)));
    }
    compute(index, k, &mut values, &mut scale, &mut valid);
    if finalize {
        let max = values.iter().map(|v| v.abs()).max().unwrap();
        let q = if max == 0 {
            0
        } else {
            ((127 - max.leading_zeros()) as i32 + scale - 34).clamp(-127, 127)
        };
        let mut result = [0u8; 48];
        for col in 0..32 {
            result[col] = rne(values[col], q - scale + 28).clamp(-127, 127) as i8 as u8;
        }
        result[32] = (q + 127) as u8;
        words[6][5..8].copy_from_slice(&pack(&result));
    } else {
        words[6][5..22].copy_from_slice(&pack(&state(
            &values,
            k / 32 + usize::from(accumulate),
            scale,
            valid,
        )));
    }
    Case {
        words,
        k,
        accumulate,
        finalize,
    }
}
#[no_mangle]
pub extern "C" fn mxint5_case_load(index: u32) {
    *CURRENT.lock().unwrap() = Some(make(index as usize));
}
#[no_mangle]
pub extern "C" fn mxint5_case_field(field: u32) -> u32 {
    let guard = CURRENT.lock().unwrap();
    let c = guard.as_ref().unwrap();
    match field {
        0 => c.k as u32,
        1 => c.accumulate as u32,
        2 => c.finalize as u32,
        _ => panic!("Invalid field"),
    }
}
#[no_mangle]
pub extern "C" fn mxint5_case_word(role: u32, row: u32, high: u32) -> u64 {
    let guard = CURRENT.lock().unwrap();
    let c = guard.as_ref().unwrap();
    (c.words[role as usize][row as usize] >> (64 * high)) as u64
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legal_cases_and_integer_rounding() {
        assert_eq!(rne(5, 1), 2);
        assert_eq!(rne(7, 1), 4);
        assert_eq!(rne(-5, 1), -2);
        assert_eq!(rne(-7, 1), -4);
        for i in 0..KS.len() {
            let c = make(i);
            assert_eq!(c.words[6][4], u128::from_le_bytes([0x5a; 16]));
            assert_eq!(
                c.words[6][if c.finalize { 8 } else { 22 }],
                u128::from_le_bytes([0x5a; 16])
            );
        }
    }
}
