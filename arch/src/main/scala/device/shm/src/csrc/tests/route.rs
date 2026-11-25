use buckyball_device_shm::Storage;

// poly's hart links (examples/designs/poly/devices/poly/arch/HartLinks.scala): a tree with hart 0 at its root.
fn poly() -> Storage {
    let links = vec![
        (0, 1),
        (1, 2),
        (0, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (0, 7),
        (7, 8),
    ];
    Storage::new(&(0..9).collect::<Vec<_>>(), 256 * 1024, links)
}

fn transfer(storage: &mut Storage, from: usize, to: usize, bytes: u64) -> (u64, u64) {
    for offset in (0..bytes).step_by(8) {
        storage.request(from, 17, offset, offset);
    }
    for (field, value) in [0, to as u64, 4096, bytes].into_iter().enumerate() {
        storage.request(from, 13, field as u64, value);
    }
    storage.request(from, 14, 0, 0)
}

#[test]
fn every_pair_is_routed() {
    let mut storage = poly();
    for from in 0..9 {
        for to in 0..9 {
            assert_eq!(
                storage.request(from, 15, 3, to as u64).0,
                u64::from(from != to)
            );
        }
    }
    assert_eq!(transfer(&mut storage, 2, 2, 16).0, 1);
}

#[test]
fn source_pipeline_overlaps_short_route_setup() {
    let mut storage = poly();
    // A beat over one link (2 routers) and over four (2 -> 1 -> 0 -> 3 -> 4, 5 routers).
    assert_eq!(transfer(&mut storage, 0, 1, 16), (0, 13));
    assert_eq!(transfer(&mut storage, 2, 4, 16), (0, 15));
    assert_eq!(storage.request(4, 16, 4096, 0), (0, 0));
    // 8 KiB from a page boundary: two bursts of 256 beats.
    assert_eq!(
        transfer(&mut storage, 8, 6, 8192),
        (0, 2 + 2 * (9 + 2 + 256 + 64))
    );
    assert_eq!(storage.request(6, 16, 4096 + 8184, 0), (8184, 0));
}

#[test]
fn four_credit_groups_and_page_boundaries() {
    let mut storage = poly();
    for offset in (0..8192).step_by(8) {
        storage.request(0, 17, offset, offset + 123);
    }
    // Source, destination, bytes, Control request-to-reply cycles.
    for (source, target, bytes, cycles) in [
        (0, 8192, 16, 13),
        (0, 8192, 48, 15),
        (0, 8192, 64, 16),
        (0, 8192, 80, 18),
        (0, 8192, 256, 31),
        (0, 8192, 4096, 331),
        (0, 8192, 4112, 342),
        (4080, 16384, 80, 27),
        (0, 12272, 80, 27),
        (4080, 12256, 80, 37),
    ] {
        for (field, value) in [source, 1, target, bytes].into_iter().enumerate() {
            storage.request(0, 13, field as u64, value);
        }
        assert_eq!(storage.request(0, 14, 0, 0), (0, cycles));
        for offset in (0..bytes).step_by(8) {
            assert_eq!(
                storage.request(1, 16, target + offset, 0).0,
                source + offset + 123
            );
        }
    }
}
