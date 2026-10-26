mod core;
use core::riscv::{
    csr::{Csrs, FS, MIE, SIE},
    Privilege,
};
use std::ffi::c_void;

#[derive(Default, Clone, Copy)]
struct Inputs {
    address: u16,
    operand: u64,
    operation: u32,
    write: bool,
    commit: bool,
    trap: bool,
    pc: u64,
    cause: u64,
    value: u64,
    returning: bool,
    return_mode: u32,
    retired: bool,
    floating: bool,
    flags: u8,
}
struct Model {
    csrs: Csrs,
    privilege: Privilege,
    satp: u64,
    address_bits: u32,
    inputs: Inputs,
    values: [u64; 18],
    read_value: u64,
    illegal: bool,
}
fn mode(value: u32) -> Privilege {
    match value {
        0 => Privilege::User,
        1 => Privilege::Supervisor,
        3 => Privilege::Machine,
        _ => panic!("Reserved privilege encoding"),
    }
}
unsafe fn model<'a>(p: *mut c_void) -> &'a mut Model {
    &mut *p.cast::<Model>()
}
#[no_mangle]
pub extern "C" fn privileged_create(address_bits: u32) -> *mut c_void {
    assert!((32..=56).contains(&address_bits));
    Box::into_raw(Box::new(Model {
        csrs: Csrs::default(),
        privilege: Privilege::Machine,
        satp: 0,
        address_bits,
        inputs: Inputs::default(),
        values: [0; 18],
        read_value: 0,
        illegal: false,
    }))
    .cast()
}
#[no_mangle]
pub unsafe extern "C" fn privileged_destroy(p: *mut c_void) {
    drop(Box::from_raw(p.cast::<Model>()));
}
#[no_mangle]
pub unsafe extern "C" fn privileged_reset(p: *mut c_void) {
    let m = model(p);
    m.csrs = Csrs::default();
    m.privilege = Privilege::Machine;
    m.satp = 0;
}
#[no_mangle]
pub unsafe extern "C" fn privileged_evaluate(
    p: *mut c_void,
    address: u32,
    operand: u64,
    operation: u32,
    write: u8,
    commit: u8,
    trap: u8,
    pc: u64,
    cause: u64,
    value: u64,
    returning: u8,
    return_mode: u32,
    hart: u64,
    interrupts: u64,
    retired: u8,
    floating: u8,
    flags: u32,
) {
    let m = model(p);
    m.inputs = Inputs {
        address: address as u16,
        operand,
        operation,
        write: write != 0,
        commit: commit != 0,
        trap: trap != 0,
        pc,
        cause,
        value,
        returning: returning != 0,
        return_mode,
        retired: retired != 0,
        floating: floating != 0,
        flags: flags as u8,
    };
    let read = match address {
        // The hardware profile exposes RV64 mstatush as zero and omits environment CSRs.
        0x310 if m.privilege == Privilege::Machine => Ok(0),
        0x10a | 0x30a => Err(core::riscv::csr::CsrError),
        _ => m
            .csrs
            .read(address as u16, m.privilege, hart, 0, interrupts),
    };
    m.illegal = read.is_err() || (write != 0 && address >> 10 == 3);
    m.read_value = read.unwrap_or(0);
    if address == 0x180 && !m.illegal {
        m.read_value = m.satp;
    }
    let pending = m.csrs.software_pending | interrupts;
    let enabled = pending & m.csrs.ie;
    let machine = m.privilege != Privilege::Machine || m.csrs.status & MIE != 0;
    let supervisor = m.privilege == Privilege::User
        || (m.privilege == Privilege::Supervisor && m.csrs.status & SIE != 0);
    let eligible = enabled
        & ((if machine { !m.csrs.ideleg } else { 0 })
            | (if supervisor { m.csrs.ideleg } else { 0 }));
    let interrupt = [11, 3, 7, 9, 1, 5]
        .into_iter()
        .find(|bit| eligible & (1 << bit) != 0);
    let delegated = m.privilege != Privilege::Machine
        && (if cause >> 63 != 0 {
            m.csrs.ideleg
        } else {
            m.csrs.edeleg
        }) & (1 << (cause & 63))
            != 0;
    let vector = m.csrs.tvec[usize::from(!delegated)];
    let redirect = if trap != 0 {
        (vector & !3).wrapping_add(if cause >> 63 != 0 && vector & 1 != 0 {
            (cause & 63) * 4
        } else {
            0
        })
    } else if return_mode == 3 {
        m.csrs.epc[1]
    } else {
        m.csrs.epc[0]
    };
    let status = m.csrs.status;
    m.values = [
        if m.illegal { 0 } else { m.read_value },
        u64::from(m.illegal),
        u64::from(address == 0xc01),
        u64::from(trap != 0 || returning != 0),
        if trap != 0 || returning != 0 {
            redirect
        } else {
            0
        },
        u64::from(interrupt.is_some()),
        interrupt.map(|bit| 1u64 << 63 | bit).unwrap_or(0),
        u64::from(enabled != 0),
        m.privilege as u64,
        if status & (1 << 17) != 0 {
            (status >> 11) & 3
        } else {
            m.privilege as u64
        },
        m.satp,
        (status >> 18) & 1,
        (status >> 19) & 1,
        u64::from(status & FS != 0),
        u64::from(m.csrs.fcsr >> 5),
        (status >> 20) & 1,
        (status >> 21) & 1,
        (status >> 22) & 1,
    ];
}
#[no_mangle]
pub unsafe extern "C" fn privileged_value(p: *mut c_void, index: u32) -> u64 {
    model(p).values[index as usize]
}
#[no_mangle]
pub unsafe extern "C" fn privileged_pmp(p: *mut c_void, index: u32, field: u32) -> u64 {
    let m = model(p);
    let address = m.csrs.pmp.address(index as usize);
    match field {
        0 => u64::from(m.csrs.pmp.config(index as usize)),
        1 => address,
        2 => (((address ^ address.wrapping_add(1)) << 2) | 3) & ((1 << m.address_bits) - 1),
        _ => panic!("Unknown PMP field"),
    }
}
#[no_mangle]
pub unsafe extern "C" fn privileged_permitted(
    p: *mut c_void,
    address: u64,
    size: u32,
    read: u8,
    write: u8,
    execute: u8,
    privilege: u32,
) -> u8 {
    let m = model(p);
    let end = u128::from(address) + (1u128 << size);
    for index in 0..16 {
        let cfg = m.csrs.pmp.config(index);
        let encoded = u128::from(m.csrs.pmp.address(index));
        let (low, high) = match (cfg >> 3) & 3 {
            0 => continue,
            1 => (
                if index == 0 {
                    0
                } else {
                    u128::from(m.csrs.pmp.address(index - 1)) * 4
                },
                encoded * 4,
            ),
            2 => (encoded * 4, encoded * 4 + 4),
            3 => {
                let bytes = (1u128 << (m.csrs.pmp.address(index).trailing_ones() + 3))
                    .min(1u128 << m.address_bits);
                let low = encoded * 4 & !(bytes - 1);
                (low, low + bytes)
            }
            _ => unreachable!(),
        };
        if low >= high || u128::from(address) >= high || end <= low {
            continue;
        }
        let access = (read == 0 || cfg & 1 != 0)
            && (write == 0 || cfg & 2 != 0)
            && (execute == 0 || cfg & 4 != 0);
        return u8::from(
            u128::from(address) >= low
                && end <= high
                && ((privilege == 3 && cfg & 0x80 == 0) || access),
        );
    }
    u8::from(privilege == 3)
}
#[no_mangle]
pub unsafe extern "C" fn privileged_advance(p: *mut c_void) {
    let m = model(p);
    let i = m.inputs;
    let old_status = m.csrs.status;
    let old_privilege = m.privilege;
    if m.csrs.inhibit & 1 == 0 {
        m.csrs.cycle = m.csrs.cycle.wrapping_add(1);
    }
    if i.retired && m.csrs.inhibit & 4 == 0 {
        m.csrs.instret = m.csrs.instret.wrapping_add(1);
    }
    if i.floating {
        m.csrs.fcsr |= i.flags & 31;
        m.csrs.status |= FS;
    }
    if i.commit && i.write && !m.illegal {
        let value = match i.operation {
            2 => m.read_value | i.operand,
            3 => m.read_value & !i.operand,
            _ => i.operand,
        };
        match i.address {
            0x180 => {
                if value >> 60 == 0 || value >> 60 == 8 {
                    m.satp = value;
                    m.csrs.satp.write(value);
                }
            }
            0x310 => (),
            address => {
                let value = if (0x3b0..=0x3bf).contains(&address) {
                    value & ((1u64 << (m.address_bits - 2)) - 1)
                } else {
                    value
                };
                m.csrs
                    .write(address, value, old_privilege)
                    .expect("Legal CSR write rejected by the reference");
            }
        }
    }
    if i.trap {
        let delegated = old_privilege != Privilege::Machine
            && (if i.cause >> 63 != 0 {
                m.csrs.ideleg
            } else {
                m.csrs.edeleg
            }) & (1 << (i.cause & 63))
                != 0;
        let slot = usize::from(!delegated);
        m.csrs.epc[slot] = i.pc & !1;
        m.csrs.cause[slot] = i.cause;
        m.csrs.tval[slot] = i.value;
        if delegated {
            m.csrs.status = old_status & !0x122
                | ((old_status >> 1) & 1) << 5
                | u64::from(old_privilege == Privilege::Supervisor) << 8;
            m.privilege = Privilege::Supervisor;
        } else {
            m.csrs.status =
                old_status & !0x1888 | ((old_status >> 3) & 1) << 7 | (old_privilege as u64) << 11;
            m.privilege = Privilege::Machine;
        }
    } else if i.returning {
        if i.return_mode == 3 {
            m.privilege = mode(((old_status >> 11) & 3) as u32);
            m.csrs.status = old_status & !0x1888 | ((old_status >> 7) & 1) << 3 | 0x80;
            if m.privilege != Privilege::Machine {
                m.csrs.status &= !(1 << 17);
            }
        } else {
            assert_eq!(i.return_mode, 1);
            m.privilege = if old_status & 0x100 != 0 {
                Privilege::Supervisor
            } else {
                Privilege::User
            };
            m.csrs.status = old_status & !0x20122 | ((old_status >> 5) & 1) << 1 | 0x20;
        }
    }
}
