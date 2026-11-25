/// The hart SharedMem behind the thread's ordinary DDR accesses.
pub struct SharedMem(pub std::sync::RwLock<buckyball_hart_shm::Storage>);

impl bebop_bemu::ThreadMemory for SharedMem {
    fn span(&self, thread: usize, address: u64, write: bool) -> (bool, u64) {
        self.0.read().unwrap().span(thread, address, write)
    }
    fn contains(&self, thread: usize, address: u64, bytes: usize, write: bool) -> bool {
        self.0.read().unwrap().contains(thread, address, bytes, write)
    }
    fn transfer_cycles(&self, thread: usize, spans: &[(u64, usize)], write: bool) -> Option<u64> {
        self.0.read().unwrap().transfer_cycles(thread, spans, write)
    }
    fn start(&self, thread: usize, task: u64) {
        self.0.write().unwrap().start(thread, task);
    }
    fn control(&self, operation: u32, field: u64, value: u64) -> u64 {
        self.0.write().unwrap().control(operation, field, value)
    }
    fn read(&self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        self.0.read().unwrap().read(thread, address, output)
    }
    fn write(&self, thread: usize, address: u64, input: &[u8]) -> bool {
        self.0.write().unwrap().write(thread, address, input)
    }
}
