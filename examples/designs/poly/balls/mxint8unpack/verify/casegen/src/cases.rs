use std::cell::RefCell;
struct Case {
    fields: [u32; 6],
    words: [Vec<u8>; 2],
}
thread_local! { static CASE: RefCell<Option<Case>> = const { RefCell::new(None) }; }

#[no_mangle]
pub extern "C" fn mxint8unpack_case_load(index: u32) {
    assert!(index < 15);
    let blocks = [1, 8, 16, 17, 33, 64, 127, 128, 1, 0, 129, 1, 1, 1, 1][index as usize];
    let reject = index >= 9;
    let mut input = vec![0xaf; 16384];
    let mut output = vec![0x5a; 16384];
    for block in 0..blocks as usize {
        let exponent = match index {
            0 => 0,
            1 => 255,
            2 => 1,
            _ => (block * 37 + index as usize * 17) % 256,
        };
        input[32 * blocks as usize + block] = exponent as u8;
        for lane in 0..32 {
            let edge = [0, -1, 1, -128, 127];
            let code = if lane < 5 {
                edge[lane]
            } else {
                ((lane * 17 + block * 7 + index as usize) % 256) as i32 - 128
            };
            input[block * 32 + lane] = code as i8 as u8;
            if !reject {
                let value = (f64::from(code) * 2.0_f64.powi(exponent as i32 - 127)) as f32;
                let at = (block * 32 + lane) * 4;
                output[at..at + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    let fields = [
        blocks,
        u32::from(index == 11),
        u32::from(index == 12),
        if index == 13 { 2 } else { 1 },
        if index == 14 { 2 } else { 1 },
        u32::from(reject),
    ];
    CASE.with(|c| {
        *c.borrow_mut() = Some(Case {
            fields,
            words: [input, output],
        })
    });
}
#[no_mangle]
pub extern "C" fn mxint8unpack_case_field(field: u32) -> u32 {
    CASE.with(|c| c.borrow().as_ref().unwrap().fields[field as usize])
}
#[no_mangle]
pub extern "C" fn mxint8unpack_case_word(role: u32, row: u32, high: u32) -> u64 {
    CASE.with(|c| {
        let c = c.borrow();
        let bytes = &c.as_ref().unwrap().words[role as usize];
        let at = row as usize * 16 + high as usize * 8;
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    })
}
