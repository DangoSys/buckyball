fn rounded(value: i64, divisor: i64) -> i64 {
    let n = (value as i128).abs();
    let divisor = divisor as i128;
    let q = n / divisor;
    let r = n % divisor;
    let q = q + i128::from(2 * r > divisor || (2 * r == divisor && q & 1 != 0));
    if value < 0 {
        -q as i64
    } else {
        q as i64
    }
}
fn clamp(value: i64) -> i32 {
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}
fn input(index: u32, lane: usize, fraction: u32) -> i32 {
    let q = 1i64 << fraction;
    let values = [
        i32::MIN as i64,
        i32::MAX as i64,
        -3 * q,
        -3 * q + 1,
        -q,
        -q / 2,
        0,
        q / 2,
        q,
        3 * q - 1,
        3 * q,
        4 * q,
        -7,
        -5,
        5,
        7,
    ];
    values[(lane + index as usize) % values.len()] as i32
}
fn other(mode: u32, lane: usize, fraction: u32) -> i32 {
    if mode <= 2 {
        if lane < 16 {
            if mode != 0 {
                1
            } else {
                [0, 1, 1 << 30, 3][lane % 4]
            }
        } else if mode != 0 {
            0
        } else {
            [0, 1, 24, 61][lane % 4]
        }
    } else {
        [
            1 << fraction,
            -(1 << fraction),
            1,
            i32::MIN,
            i32::MAX,
            0,
            3,
            -5,
        ][lane % 8]
    }
}
#[no_mangle]
pub extern "C" fn fixed_case_word(index: u32, bank: u32, row: u32, half_word: u32) -> u64 {
    assert!(index < 108 && bank < 3 && row < 16 && half_word < 2);
    if index >= 72 && index < 90 {
        let case = (index - 72) % 9;
        let fraction = [1, 8, 24][(case % 3) as usize];
        if case < 6 {
            if bank != 0 {
                let channel = (row * 2 + half_word) as usize;
                let previous = (channel as i64 - 8) * (1i64 << 33);
                let value = if bank == 1 {
                    previous
                } else {
                    previous + (case / 3 + 1) as i64 * input(index, channel, fraction) as i64
                };
                return value as u64;
            }
            let lane = (row * 4 + half_word * 2) as usize;
            return input(index, lane, fraction) as u32 as u64
                | ((input(index, lane + 1, fraction) as u32 as u64) << 32);
        }
        let sums = [
            1,
            3,
            -1,
            -3,
            5,
            7,
            -5,
            -7,
            i32::MAX as i64 * 3136,
            i32::MIN as i64 * 3136,
            100,
            -100,
            i64::MAX,
            i64::MIN,
            0,
            -49,
        ];
        let divisors = [2, 2, 2, 2, 2, 2, 2, 2, 3136, 3136, 49, 49, 1, 1, 196, 49];
        let rotation = ((index - 72) / 9) as usize;
        if bank == 0 {
            return sums[((row * 2 + half_word) as usize + rotation) % 16] as u64;
        }
        let mut packed = 0u64;
        for half_lane in 0..2 {
            let channel = ((row * 4 + half_word * 2 + half_lane) as usize + rotation) % 16;
            let value = if bank == 1 {
                divisors[channel] as i32
            } else {
                clamp(rounded(sums[channel], divisors[channel]))
            };
            packed |= (value as u32 as u64) << (half_lane * 32);
        }
        return packed;
    }
    let stream = index >= 90;
    let mode = if stream { (index - 90) % 3 } else { index % 6 };
    let fraction = [1, 8, 24][if stream {
        ((index - 90) / 3) % 3
    } else {
        (index / 6) % 3
    } as usize];
    let broadcast = !stream && mode == 3 && (index / 18) % 2 == 1;
    let mut packed = 0u64;
    for half_lane in 0..2 {
        let lane = (row * 4 + half_word * 2 + half_lane) as usize;
        let a = input(index, lane, fraction);
        let channel = if stream && bank == 1 {
            (row / 2 * 4 + half_word * 2 + half_lane) as usize
        } else {
            lane
        };
        let multiplier = (1 + channel % 5) as i32;
        let shift = (channel % 7) as i32;
        let result = match bank {
            0 => a,
            1 => {
                if stream {
                    if row % 2 == 0 {
                        multiplier
                    } else {
                        shift
                    }
                } else {
                    other(mode, lane, fraction)
                }
            }
            2 => {
                let unit = 1i64 << fraction;
                let q = if mode <= 2 {
                    clamp(rounded(
                        i64::from(a)
                            * i64::from(if stream {
                                multiplier
                            } else {
                                other(mode, lane % 16, fraction)
                            }),
                        1i64 << if stream {
                            shift
                        } else {
                            other(mode, 16 + lane % 16, fraction)
                        },
                    )) as i64
                } else {
                    a as i64
                };
                let b = other(mode, if broadcast { lane % 16 } else { lane }, fraction) as i64;
                clamp(match mode {
                    0 => q,
                    1 => rounded(q * (q + 3 * unit).clamp(0, 6 * unit), 6 * unit),
                    2 => rounded((q + 3 * unit).clamp(0, 6 * unit), 6),
                    3 => rounded(q * b, unit),
                    4 => q.max(0),
                    5 => q + b,
                    _ => unreachable!(),
                })
            }
            _ => unreachable!(),
        };
        packed |= (result as u32 as u64) << (half_lane * 32);
    }
    packed
}
