mod vectors;
type Input = (u32, u64, u64, u64, u64, u32);
type Output = (u64, u32, bool, bool);

#[no_mangle]
pub extern "C" fn fpu_case_count() -> u32 {
    vectors::CASES.len() as u32
}

#[no_mangle]
pub extern "C" fn fpu_cancel_case() -> u32 {
    vectors::CASES
        .iter()
        .position(|(input, _)| input.0 & 0x7f == 0x53 && input.0 >> 27 == 3)
        .unwrap() as u32
}

#[no_mangle]
pub unsafe extern "C" fn fpu_case(
    index: u32,
    instruction: *mut u32,
    integer: *mut u64,
    a: *mut u64,
    rhs: *mut u64,
    c: *mut u64,
    rounding: *mut u32,
) {
    let input = vectors::CASES[index as usize].0;
    *instruction = input.0;
    *integer = input.1;
    *a = input.2;
    *rhs = input.3;
    *c = input.4;
    *rounding = input.5;
}

#[no_mangle]
pub unsafe extern "C" fn fpu_expected(
    instruction: u32,
    integer: u64,
    a: u64,
    rhs: u64,
    c: u64,
    rounding: u32,
    data: *mut u64,
    flags: *mut u32,
    is_integer: *mut u8,
    illegal: *mut u8,
) {
    let observed = (instruction, integer, a, rhs, c, rounding);
    let Some((_, expected)) = vectors::CASES.iter().find(|(input, _)| *input == observed) else {
        panic!("No reference for observed FPU request: {observed:?}");
    };
    *data = expected.0;
    *flags = expected.1;
    *is_integer = u8::from(expected.2);
    *illegal = u8::from(expected.3);
}
