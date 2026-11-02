/// The core SharedMem behind the thread's ordinary DDR accesses.
pub struct SharedMem(pub std::sync::Mutex<buckyball_core_shm::Storage>);

impl bebop_bemu::core::ThreadMemory for SharedMem {
    fn contains(&self, thread: usize, address: u64, bytes: usize, write: bool) -> bool {
        self.0.lock().unwrap().contains(thread, address, bytes, write)
    }
    fn transfer_cycles(&self, thread: usize, address: u64, rows: u64, write: bool) -> Option<u64> {
        self.0
            .lock()
            .unwrap()
            .transfer_cycles(thread, address, rows, write)
    }
    fn start(&self, thread: usize, task: u64) {
        self.0.lock().unwrap().start(thread, task);
    }
    fn control(&self, operation: u32, field: u64, value: u64) -> u64 {
        self.0.lock().unwrap().control(operation, field, value)
    }
    fn read(&self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        self.0.lock().unwrap().read(thread, address, output)
    }
    fn write(&self, thread: usize, address: u64, input: &[u8]) -> bool {
        self.0.lock().unwrap().write(thread, address, input)
    }
}
