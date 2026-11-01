// Ant instruction classes from the RV64IM encodings, numbered as the RTL `Kind` enum:
// illegal, lui, auipc, jal, jalr, branch, load, store, alu, multiply, divide, custom,
// fence, exit, breakpoint, cycle. The only CSR access is `csrr rd, mcycle`.
#[no_mangle]
pub extern "C" fn ant_decode_kind(i: u32) -> u32 {
    let f3 = (i >> 12) & 7;
    let f7 = i >> 25;
    match i & 0x7f {
        0x37 => 1,
        0x17 => 2,
        0x6f => 3,
        0x67 if f3 == 0 => 4,
        0x63 if f3 != 2 && f3 != 3 => 5,
        0x03 if f3 != 7 => 6,
        0x23 if f3 < 4 => 7,
        0x13 => match f3 {
            1 if f7 >> 1 == 0 => 8,
            5 if f7 >> 1 == 0 || f7 >> 1 == 0x10 => 8,
            1 | 5 => 0,
            _ => 8,
        },
        0x1b => match f3 {
            0 => 8,
            1 if f7 == 0 => 8,
            5 if f7 == 0 || f7 == 0x20 => 8,
            _ => 0,
        },
        0x33 => match f7 {
            0 => 8,
            0x20 if f3 == 0 || f3 == 5 => 8,
            1 if f3 < 4 => 9,
            1 => 10,
            _ => 0,
        },
        0x3b => match f7 {
            0 if f3 == 0 || f3 == 1 || f3 == 5 => 8,
            0x20 if f3 == 0 || f3 == 5 => 8,
            1 if f3 == 0 => 9,
            1 if f3 >= 4 => 10,
            _ => 0,
        },
        0x7b => 11,
        0x0f if f3 == 0 => 12,
        0x73 if i == 0x0000_0073 => 13,
        0x73 if i == 0x0010_0073 => 14,
        0x73 if i & 0xffff_f07f == 0xb000_2073 => 15,
        _ => 0,
    }
}
