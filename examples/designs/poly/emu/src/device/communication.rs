use std::sync::{Arc, Mutex};

struct Communication(Mutex<buckyball_device_shm::Storage>);
impl bebop_bemu::device::Service for Communication {
    fn request(&self, core: usize, operation: u32, address: u64, data: u64) -> u64 {
        self.0
            .lock()
            .expect("Device communication poisoned")
            .request(core, operation, address, data)
    }
}

pub fn instantiate(
    device: &bebop_bemu::DeviceModel,
    args: &bebop_bemu::sim::run::Args,
) -> Result<Vec<Box<dyn bebop_bemu::CoreModel>>, String> {
    let cores = bebop_bemu::hart_ids();
    assert!(cores.len() >= 2 && cores.iter().copied().eq(0..cores.len()));
    // The core links route between core 0 and every other core (arch/device/CoreLinks.scala).
    let links = (1..cores.len()).flat_map(|core| [(0, core), (core, 0)]).collect();
    device.set_service(Arc::new(Communication(Mutex::new(
        buckyball_device_shm::Storage::new(&cores, 256 * 1024, links),
    ))));
    bebop_bemu::device::cbus::instantiate(device, args)
}
