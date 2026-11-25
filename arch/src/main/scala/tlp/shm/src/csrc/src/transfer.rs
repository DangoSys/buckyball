#[no_mangle]
pub extern "C" fn mesh_route(destination: u32, row: u32, col: u32) -> u32 {
    let target_row = destination >> 2;
    let target_col = destination & 3;
    match (target_col.cmp(&col), target_row.cmp(&row)) {
        (std::cmp::Ordering::Greater, _) => 1,
        (std::cmp::Ordering::Less, _) => 2,
        (_, std::cmp::Ordering::Greater) => 3,
        (_, std::cmp::Ordering::Less) => 4,
        _ => 0,
    }
}

struct Banks { data: Vec<u128>, initialized: Vec<u16> }
#[no_mangle]
pub extern "C" fn mesh_banks_create() -> *mut Banks {
    Box::into_raw(Box::new(Banks { data: vec![0; 12*1024], initialized: vec![0;12*1024] }))
}
#[no_mangle]
pub unsafe extern "C" fn mesh_banks_destroy(model: *mut Banks) { drop(Box::from_raw(model)); }
#[no_mangle]
pub unsafe extern "C" fn mesh_bank_access(model: *mut Banks, bank: u32, row: u32, write: u8,
                                          data: *const u32, mask: u32, output: *mut u32) -> u32 {
    let result = if bank >= 12 || row >= 1024 {
        for i in 0..4 { *output.add(i) = 0; }
        return 1;
    } else {
        let m = &mut *model;
        let index = bank as usize * 1024 + row as usize;
        if write != 0 {
            let value = (0..4).fold(0u128, |v,i| v | ((*data.add(i) as u128) << (32*i)));
            for byte in 0..16 {
                if mask & (1 << byte) != 0 {
                    let bits = 255u128 << (8*byte);
                    m.data[index] = (m.data[index] & !bits) | (value & bits);
                }
            }
            m.initialized[index] |= mask as u16;
            0
        } else {
            assert_eq!(m.initialized[index], 65535, "read of an uninitialized row");
            m.data[index]
        }
    };
    for i in 0..4 { *output.add(i) = (result >> (32*i)) as u32; }
    0
}
