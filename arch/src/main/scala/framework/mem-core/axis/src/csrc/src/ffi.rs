use std::ffi::c_void;

use crate::arbiter::ArbiterModel;
use crate::model::{AxiSModel, Beat};

#[no_mangle]
pub extern "C" fn axis_arbiter_create() -> *mut c_void {
    Box::into_raw(Box::new(ArbiterModel::default())).cast()
}

#[no_mangle]
pub unsafe extern "C" fn axis_arbiter_destroy(model: *mut c_void) {
    assert!(!model.is_null());
    drop(Box::from_raw(model.cast::<ArbiterModel>()));
}

#[no_mangle]
pub unsafe extern "C" fn axis_arbiter_step(
    model: *mut c_void,
    valid: u32,
    ready: u8,
    last: u32,
) -> u32 {
    assert!(!model.is_null());
    (*model.cast::<ArbiterModel>())
        .step(valid, ready != 0, last)
        .unwrap_or(u32::MAX)
}

#[no_mangle]
pub extern "C" fn axis_ref_create() -> *mut c_void {
    Box::into_raw(Box::new(AxiSModel::default())).cast()
}

#[no_mangle]
pub unsafe extern "C" fn axis_ref_destroy(model: *mut c_void) {
    assert!(!model.is_null());
    drop(Box::from_raw(model.cast::<AxiSModel>()));
}

#[no_mangle]
pub unsafe extern "C" fn axis_ref_push(model: *mut c_void, data: u32, keep: u8, last: u8) {
    assert!(!model.is_null());
    (*model.cast::<AxiSModel>()).push(Beat {
        data,
        keep,
        last: last != 0,
    });
}

#[no_mangle]
pub unsafe extern "C" fn axis_ref_pop(
    model: *mut c_void,
    data: *mut u32,
    keep: *mut u8,
    last: *mut u8,
) -> u8 {
    assert!(!model.is_null());
    assert!(!data.is_null());
    assert!(!keep.is_null());
    assert!(!last.is_null());

    let Some(beat) = (*model.cast::<AxiSModel>()).pop() else {
        return 0;
    };
    *data = beat.data;
    *keep = beat.keep;
    *last = u8::from(beat.last);
    1
}
