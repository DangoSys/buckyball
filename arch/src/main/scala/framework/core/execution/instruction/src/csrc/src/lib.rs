mod core;

fn signed(value: u32, width: u32) -> u64 {
    ((value << (32 - width)) as i32 >> (32 - width)) as i64 as u64
}

#[no_mangle]
pub unsafe extern "C" fn instruction_expected(
    bits: u32,
    expanded: *mut u32,
    illegal: *mut u8,
    immediate: *mut u64,
    atomic: *mut u32,
    function: *mut u32,
    word: *mut u8,
    immediate_mode: *mut u8,
) {
    let insn = if bits & 3 != 3 {
        match core::riscv::expand_compressed(bits as u16) {
            Ok(value) => value,
            Err(_) => {
                *expanded = 0;
                *illegal = 1;
                *immediate = 0;
                *atomic = 0;
                *function = 0;
                *word = 0;
                *immediate_mode = 0;
                return;
            }
        }
    } else {
        bits
    };
    let opcode = insn & 127;
    let funct3 = (insn >> 12) & 7;
    let funct7 = insn >> 25;
    let rd = (insn >> 7) & 31;
    let rs2 = (insn >> 20) & 31;
    let is_word = matches!(opcode, 0x1b | 0x3b);
    let atomic_code = match funct7 >> 2 {
        0 => 2,
        1 => 1,
        2 => 10,
        3 => 11,
        4 => 3,
        8 => 5,
        12 => 4,
        16 => 6,
        20 => 7,
        24 => 8,
        28 => 9,
        _ => 0,
    };
    let legal = match opcode {
        0x37 | 0x17 | 0x6f => true,
        0x67 => funct3 == 0,
        0x63 => matches!(funct3, 0 | 1 | 4 | 5 | 6 | 7),
        0x13 | 0x1b => match funct3 {
            0 => true,
            1 => {
                if is_word {
                    funct7 == 0
                } else {
                    funct7 >> 1 == 0
                }
            }
            5 => {
                if is_word {
                    matches!(funct7, 0 | 32)
                } else {
                    matches!(funct7 >> 1, 0 | 16)
                }
            }
            _ => !is_word,
        },
        0x33 | 0x3b => {
            if funct7 == 1 {
                !is_word || funct3 == 0 || funct3 >= 4
            } else {
                (funct7 == 0 || (funct7 == 32 && matches!(funct3, 0 | 5)))
                    && (!is_word || matches!(funct3, 0 | 1 | 5))
            }
        }
        0x03 => funct3 <= 6,
        0x23 => funct3 <= 3,
        0x07 | 0x27 => matches!(funct3, 2 | 3),
        0x2f => matches!(funct3, 2 | 3) && atomic_code != 0 && (funct7 >> 2 != 2 || rs2 == 0),
        0x43 | 0x47 | 0x4b | 0x4f | 0x53 => true,
        0x0f => matches!(funct3, 0 | 1),
        0x73 => {
            if funct3 == 0 {
                matches!(
                    insn,
                    0x00000073 | 0x00100073 | 0x30200073 | 0x10200073 | 0x10500073
                ) || (funct7 == 9 && rd == 0)
            } else {
                funct3 != 4
            }
        }
        0x2b => funct3 == 7,
        0x7b => true,
        _ => false,
    };
    *expanded = insn;
    *illegal = u8::from(!legal);
    *immediate = match opcode {
        0x23 | 0x27 => signed(((insn >> 25) << 5) | ((insn >> 7) & 31), 12),
        0x63 => signed(
            ((insn >> 31) << 12)
                | (((insn >> 7) & 1) << 11)
                | (((insn >> 25) & 63) << 5)
                | (((insn >> 8) & 15) << 1),
            13,
        ),
        0x6f => signed(
            ((insn >> 31) << 20)
                | (((insn >> 12) & 255) << 12)
                | (((insn >> 20) & 1) << 11)
                | (((insn >> 21) & 1023) << 1),
            21,
        ),
        0x17 | 0x37 => signed(insn & 0xfffff000, 32),
        _ => signed(insn >> 20, 12),
    };
    *atomic = atomic_code;
    *function = match funct3 {
        0 => {
            if matches!(opcode, 0x33 | 0x3b) && funct7 & 32 != 0 {
                10
            } else {
                0
            }
        }
        1 => 1,
        2 => 12,
        3 => 14,
        4 => 4,
        5 => {
            if funct7 & 32 != 0 {
                11
            } else {
                5
            }
        }
        6 => 6,
        7 => 7,
        _ => unreachable!(),
    };
    *word = u8::from(is_word);
    *immediate_mode = u8::from(matches!(opcode, 0x13 | 0x1b));
}
