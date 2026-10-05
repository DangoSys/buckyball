use std::collections::BTreeMap;
use std::ffi::c_void;

struct Range {
    va: u64,
    pa: u64,
    bytes: u32,
    write: bool,
}
struct Entry {
    ranges: Vec<Range>,
    terminal: bool,
    notified: bool,
    error: u32,
    fault: u64,
}
struct Map {
    entries: BTreeMap<u32, Entry>,
    address_bits: u32,
}

#[no_mangle]
pub extern "C" fn pmap_ref_create(address_bits: u32) -> *mut c_void {
    Box::into_raw(Box::new(Map {
        entries: BTreeMap::new(),
        address_bits,
    }))
    .cast()
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_destroy(ptr: *mut c_void) {
    drop(Box::from_raw(ptr.cast::<Map>()));
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_reset(ptr: *mut c_void) {
    (*ptr.cast::<Map>()).entries.clear();
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_reserve(ptr: *mut c_void, id: u32) {
    let map = &mut *ptr.cast::<Map>();
    assert!(map.entries.len() < 4 && !map.entries.contains_key(&id) && id < 256);
    map.entries.insert(
        id,
        Entry {
            ranges: Vec::new(),
            terminal: false,
            notified: false,
            error: 0,
            fault: 0,
        },
    );
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_prepared(
    ptr: *mut c_void,
    id: u32,
    va: u64,
    pa: u64,
    bytes: u32,
    write: u32,
    last: u32,
    error: u32,
) {
    let map = &mut *ptr.cast::<Map>();
    let entry = map.entries.get_mut(&id).expect("reserved prepared ID");
    assert!(!entry.terminal);
    if error != 0 {
        assert!(last != 0);
        entry.ranges.clear();
        entry.error = error;
        entry.fault = va;
    } else {
        assert!(bytes > 0 && bytes <= 4096 && entry.ranges.len() < 8);
        assert!((va & 4095) + u64::from(bytes) <= 4096 && (pa & 4095) + u64::from(bytes) <= 4096);
        assert!(va.checked_add(u64::from(bytes - 1)).is_some());
        assert!((pa as u128 + bytes as u128 - 1) < (1u128 << map.address_bits));
        assert!(entry
            .ranges
            .first()
            .map_or(true, |r| r.write == (write != 0)));
        entry.ranges.push(Range {
            va,
            pa,
            bytes,
            write: write != 0,
        });
        assert!(entry.ranges.len() != 8 || last != 0);
    }
    entry.terminal = last != 0;
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_ready(
    ptr: *mut c_void,
    id: u32,
    fire: u32,
    error: *mut u32,
    va: *mut u64,
) -> u32 {
    let map = &mut *ptr.cast::<Map>();
    let Some(entry) = map.entries.get_mut(&id) else {
        return 0;
    };
    if !entry.terminal || entry.notified {
        return 0;
    }
    *error = entry.error;
    *va = entry.fault;
    if fire != 0 {
        entry.notified = true;
    }
    1
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_query(
    ptr: *mut c_void,
    valid: u32,
    id: u32,
    va: u64,
    bytes: u32,
    write: u32,
    retiring: u32,
    retiring_id: u32,
    hit: *mut u32,
    pa: *mut u64,
    error: *mut u32,
) {
    *hit = 0;
    *pa = 0;
    *error = 0;
    if valid == 0 {
        return;
    }
    let Some(last) = va.checked_add(u64::from(bytes).wrapping_sub(1)) else {
        *error = 1;
        return;
    };
    if bytes == 0 || (va & 4095) + u64::from(bytes) > 4096 {
        *error = 1;
        return;
    }
    let map = &*ptr.cast::<Map>();
    if retiring != 0 && retiring_id == id {
        *error = 4;
        return;
    }
    if let Some(entry) = map.entries.get(&id) {
        if entry.notified && entry.error == 0 {
            for range in &entry.ranges {
                if range.write == (write != 0)
                    && va >= range.va
                    && (last as u128) < range.va as u128 + range.bytes as u128
                {
                    let translated = range.pa + (va - range.va);
                    assert!(
                        *hit == 0 || *pa == translated,
                        "inconsistent overlapping mappings"
                    );
                    *hit = 1;
                    *pa = translated;
                }
            }
        }
    }
    if *hit == 0 {
        *error = 4;
    }
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_release(ptr: *mut c_void, id: u32) {
    let entry = (*ptr.cast::<Map>())
        .entries
        .remove(&id)
        .expect("reserved release ID");
    assert!(entry.notified);
}
#[no_mangle]
pub unsafe extern "C" fn pmap_ref_pending(ptr: *mut c_void) -> u32 {
    (*ptr.cast::<Map>()).entries.len() as u32
}
