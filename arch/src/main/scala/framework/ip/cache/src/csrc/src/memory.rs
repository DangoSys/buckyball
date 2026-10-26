use std::ffi::c_void;

#[derive(Clone)]
struct Entry {
    valid: bool,
    addr: u64,
    data: Vec<u8>,
    metadata: u32,
}
struct Cache {
    lines: Vec<Vec<Entry>>,
    next: Vec<usize>,
    line_bytes: usize,
}

#[no_mangle]
pub extern "C" fn cache_ref_create(sets: u32, ways: u32, line_bytes: u32) -> *mut c_void {
    let empty = Entry {
        valid: false,
        addr: 0,
        data: vec![0; line_bytes as usize],
        metadata: 0,
    };
    Box::into_raw(Box::new(Cache {
        lines: vec![vec![empty; ways as usize]; sets as usize],
        next: vec![0; sets as usize],
        line_bytes: line_bytes as usize,
    }))
    .cast()
}

#[no_mangle]
pub unsafe extern "C" fn cache_ref_destroy(model: *mut c_void) {
    drop(Box::from_raw(model.cast::<Cache>()));
}

#[no_mangle]
pub unsafe extern "C" fn cache_ref_reset(model: *mut c_void) {
    let model = &mut *model.cast::<Cache>();
    for set in &mut model.lines {
        for entry in set {
            entry.valid = false;
        }
    }
    model.next.fill(0);
}

#[no_mangle]
pub unsafe extern "C" fn cache_ref_access(
    model: *mut c_void,
    op: u32,
    addr: u64,
    way: u32,
    data: *const u32,
    mask: *const u32,
    metadata: u32,
    eligible: *const u32,
    hit_out: *mut u8,
    available_out: *mut u8,
    way_out: *mut u32,
    valid_out: *mut u8,
    addr_out: *mut u64,
    data_out: *mut u32,
    metadata_out: *mut u32,
) {
    let model = &mut *model.cast::<Cache>();
    assert!(op <= 4 && addr % model.line_bytes as u64 == 0);
    let index = (addr as usize / model.line_bytes) % model.lines.len();
    let set = &mut model.lines[index];
    let hit = set.iter().position(|e| e.valid && e.addr == addr);
    let eligible = std::slice::from_raw_parts(eligible, set.len().div_ceil(32));
    let allowed = |w: usize| eligible[w / 32] & (1 << (w % 32)) != 0;
    let selected = if op == 0 {
        hit.or_else(|| {
            set.iter()
                .enumerate()
                .position(|(w, e)| !e.valid && allowed(w))
        })
        .or_else(|| {
            (0..set.len())
                .map(|i| (model.next[index] + i) % set.len())
                .find(|&w| allowed(w))
        })
    } else {
        assert!((way as usize) < set.len());
        Some(way as usize)
    };
    let output = std::slice::from_raw_parts_mut(data_out, model.line_bytes / 4);
    output.fill(0);
    *hit_out = u8::from(op == 0 && hit.is_some());
    *available_out = u8::from(selected.is_some());
    *way_out = 0;
    *valid_out = 0;
    *addr_out = 0;
    *metadata_out = 0;
    let Some(w) = selected else {
        return;
    };
    *way_out = w as u32;
    let ways = set.len();
    let entry = &mut set[w];
    match op {
        0 | 1 => {
            if entry.valid {
                for (out, bytes) in output.iter_mut().zip(entry.data.chunks_exact(4)) {
                    *out = u32::from_le_bytes(bytes.try_into().unwrap());
                }
            }
        }
        2 | 3 => {
            if op == 2 {
                assert!(entry.valid && entry.addr == addr);
            }
            if op == 3 {
                assert!(hit.is_none() || hit == Some(w));
                entry.valid = true;
                entry.addr = addr;
                model.next[index] = (w + 1) % ways;
            }
            let input = std::slice::from_raw_parts(data, model.line_bytes / 4);
            let mask = std::slice::from_raw_parts(mask, model.line_bytes.div_ceil(32));
            for (i, byte) in entry.data.iter_mut().enumerate() {
                if op == 3 || mask[i / 32] & (1 << (i % 32)) != 0 {
                    *byte = (input[i / 4] >> (8 * (i % 4))) as u8;
                }
            }
            entry.metadata = metadata;
        }
        4 => {
            entry.valid = false;
        }
        _ => unreachable!(),
    }
    if entry.valid {
        *valid_out = 1;
        *addr_out = entry.addr;
        *metadata_out = entry.metadata;
    }
}
