use std::sync::{Arc, Mutex};

struct Communication {
    storage: Mutex<buckyball_device_shm::Storage>,
    offsets: Mutex<Vec<Option<u64>>>,
}
impl bebop_bemu::Service for Communication {
    fn request(&self, hart: usize, operation: u32, address: u64, data: u64) -> (u64, u64) {
        if operation == 18 {
            assert_eq!(address, 0);
            assert!(data % 8 == 0);
            assert!(self.offsets.lock().unwrap()[hart].replace(data).is_none());
            return (0, 0);
        }
        self.storage
            .lock()
            .expect("Device communication poisoned")
            .request(hart, operation, address, data)
    }
    fn exchange(&self, hart: usize, to_device: bool, bytes: &mut [u8]) -> bebop_bemu::ServicePhases {
        let offset = self.offsets.lock().unwrap()[hart].take().expect("No payload offset");
        let mut storage = self.storage.lock().unwrap();
        for (i, chunk) in bytes.chunks_mut(8).enumerate() {
            let address = offset + (i * 8) as u64;
            if to_device {
                let mut word = [0; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                storage.request(hart, 17, address, u64::from_le_bytes(word));
            } else {
                let (word, _) = storage.request(hart, 16, address, 0);
                chunk.copy_from_slice(&word.to_le_bytes()[..chunk.len()]);
            }
        }
        let beat = if offset % 16 == 0 { 16 } else { 8 };
        if to_device {
            bebop_bemu::ServicePhases {
                // ServiceTransfer launch and ReadTransport shape/plan reach AR in seven cycles.
                prepare: 7,
                // Payload has four credits; its SRAM writes overlap the incoming DMA stream.
                stream: bytes.len().div_ceil(beat) as u64,
                // Final R -> send queue -> SRAM acknowledgement -> service reply: seven cycles.
                retire: 7,
            }
        } else {
            // Pull output-queue backpressure is not calibrated. Keep its conservative serial
            // estimate separate from the qualified stage path; this is not physical DDR time.
            bebop_bemu::ServicePhases {
                prepare: 0,
                stream: bytes.len().div_ceil(beat) as u64 * 4,
                retire: 0,
            }
        }
    }
}

pub fn instantiate(
    device: &bebop_bemu::DeviceModel,
    args: &bebop_bemu::sim::run::Args,
) -> Result<Vec<Box<dyn bebop_bemu::HartModel>>, String> {
    let harts = bebop_bemu::hart_ids();
    assert!(harts.len() >= 2 && harts.iter().copied().eq(0..harts.len()));
    // The kept links of the hart mesh (arch/device/HartLinks.scala): hart 0 to 1, 3, 5, 7 and each of those to the
    // next hart round the edge, a tree the device SHM network routes every pair of harts over.
    let links = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6), (0, 7), (7, 8)]
        .into_iter()
        .filter(|&(a, b)| a < harts.len() && b < harts.len())
        .collect();
    device.link.set_service(Arc::new(Communication {
        storage: Mutex::new(buckyball_device_shm::Storage::new(&harts, 256 * 1024, links)),
        offsets: Mutex::new(vec![None; harts.len()]),
    }));
    bebop_bemu::sim::device_system::instantiate(device, args)
}
