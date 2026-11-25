#[derive(Clone)]
struct Edge {
    source: usize,
    source_task: u64,
    source_address: u64,
    target: usize,
    target_task: u64,
    target_address: u64,
    bytes: usize,
    bank: usize,
    published: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_ranges_join_and_split() {
        let mut storage = Storage::new(3, 8, 64, Mesh {
            columns: 2, thread_nodes: vec![0, 1, 3], banks_per_node: 2,
        });
        let mut tokens = Vec::new();
        for descriptor in [[0, 1, 0x1000, 2, 3, 0x3000, 48],
                           [1, 2, 0x2000, 2, 3, 0x3030, 80],
                           [2, 3, 0x4000, 0, 4, 0x5000, 48],
                           [2, 3, 0x4030, 1, 5, 0x6000, 80]] {
            for (field, value) in descriptor.into_iter().enumerate() {
                storage.control(20, field as u64, value);
            }
            tokens.push(storage.control(21, 0, 0));
        }
        storage.start(0, 1); storage.start(1, 2); storage.start(2, 3);
        let input = (0..128).map(|i| (i * 17) as u8).collect::<Vec<_>>();
        assert!(storage.write(0, 0x1000, &input[..48]));
        assert!(storage.write(1, 0x2000, &input[48..]));
        storage.control(23, tokens[0], 0); storage.control(23, tokens[1], 0);
        assert!(storage.contains(2, 0x3000, 128, false));
        let mut result = vec![0; 128];
        assert!(storage.read(2, 0x3000, &mut result));
        assert_eq!(result, input);
        assert!(storage.contains(2, 0x4000, 128, true));
        assert!(storage.write(2, 0x4000, &input));
        storage.control(23, tokens[2], 0); storage.control(23, tokens[3], 0);
        storage.start(0, 4); storage.start(1, 5);
        assert!(storage.read(0, 0x5000, &mut result[..48]));
        assert!(storage.read(1, 0x6000, &mut result[48..]));
        assert_eq!(result, input);
        assert!(storage.transfer_cycles(2, &[(0x3000, 128)], false).unwrap() > 16 + 48);
    }

    #[test]
    fn subword_at_logical_end() {
        let mut storage = Storage::new(2, 16, 64, Mesh {
            columns: 2, thread_nodes: vec![0, 1], banks_per_node: 2,
        });
        for (field, value) in [0, 1, 0x1000, 1, 2, 0x2000, 528].into_iter().enumerate() {
            storage.control(20, field as u64, value);
        }
        storage.control(21, 0, 0);
        storage.start(0, 1); storage.start(1, 2);
        assert!(storage.contains(1, 0x220c, 4, false));
        assert!(storage.transfer_cycles(1, &[(0x220c, 4)], false).is_some());
        for (field, value) in [0, 1, 0x3000, 1, 2, 0x4000, 16].into_iter().enumerate() {
            storage.control(20, field as u64, value);
        }
        storage.control(21, 0, 0);
        let spans = [(0x220c, 4), (0x400c, 4)];
        assert!(storage.transfer_cycles(1, &spans, false).is_some());
        assert!(storage.transfer_cycles(1, &[], false).is_none());
        assert!(std::panic::catch_unwind(|| {
            storage.transfer_cycles(1, &[(0x220c, 5)], false)
        }).is_err());
    }

    #[test]
    fn multibank_fanout_and_reuse() {
        let mut storage = Storage::new(3, 8, 64, Mesh {
            columns: 2, thread_nodes: vec![0, 1, 3], banks_per_node: 2,
        });
        let mut tokens = Vec::new();
        for descriptor in [[0, 1, 0x1000, 1, 2, 0x2000, 160],
                           [0, 1, 0x1000, 2, 3, 0x3000, 160]] {
            for (field, value) in descriptor.into_iter().enumerate() {
                storage.control(20, field as u64, value);
            }
            tokens.push(storage.control(21, 0, 0));
        }
        storage.start(0, 1);
        storage.start(1, 2);
        storage.start(2, 3);
        let input = (0..160).map(|i| (i * 17) as u8).collect::<Vec<_>>();
        assert!(storage.write(0, 0x1000, &input[..48]));
        assert!(storage.write(0, 0x1030, &input[48..]));
        storage.control(23, tokens[0], 0);
        let mut result = vec![0; 160];
        assert!(storage.read(1, 0x2000, &mut result[..48]));
        assert!(storage.read(1, 0x2030, &mut result[48..]));
        assert_eq!(result, input);
        storage.control(22, tokens[0], 0);
        assert!(storage.read(2, 0x3000, &mut result));
        assert_eq!(result, input);
        for (field, value) in [1, 4, 0x4000, 2, 5, 0x5000, 128].into_iter().enumerate() {
            storage.control(20, field as u64, value);
        }
        let next = storage.control(21, 0, 0);
        assert_eq!(storage.edges[next as usize].as_ref().unwrap().bank, 3);
        storage.control(22, tokens[1], 0);
        for (field, value) in [0, 6, 0x6000, 1, 7, 0x7000, 160].into_iter().enumerate() {
            storage.control(20, field as u64, value);
        }
        let reused = storage.control(21, 0, 0);
        assert_eq!(storage.edges[reused as usize].as_ref().unwrap().bank, 0);
        assert!(storage.initialized[..3].iter().flatten().all(|&valid| !valid));
    }
}

/// Hart-local data edges. This model has no device communication state.
pub struct Mesh {
    pub columns: usize,
    pub thread_nodes: Vec<usize>,
    pub banks_per_node: usize,
}

pub struct Storage {
    mesh: Mesh,
    tasks: Vec<u64>,
    edges: Vec<Option<Edge>>,
    /// The edges each thread produces and consumes: an access looks only at its own thread's.
    sources: Vec<Vec<usize>>,
    targets: Vec<Vec<usize>>,
    data: Vec<Vec<u8>>,
    initialized: Vec<Vec<bool>>,
    descriptor: [u64; 7],
    fields: u8,
}

impl Storage {
    pub fn new(threads: usize, banks: usize, bytes_per_bank: usize, mesh: Mesh) -> Self {
        assert!(threads > 0 && banks > 0 && bytes_per_bank >= 16);
        Self {
            mesh,
            tasks: vec![0; threads],
            edges: vec![None; banks],
            sources: vec![Vec::new(); threads],
            targets: vec![Vec::new(); threads],
            data: vec![vec![0; bytes_per_bank]; banks],
            initialized: vec![vec![false; bytes_per_bank]; banks],
            descriptor: [0; 7],
            fields: 0,
        }
    }

    pub fn start(&mut self, thread: usize, task: u64) {
        self.tasks[thread] = task;
    }

    fn edge(&self, thread: usize, address: u64, write: bool) -> Option<&Edge> {
        let slots = if write { &self.sources[thread] } else { &self.targets[thread] };
        let mut matches = slots.iter().map(|&slot| self.edges[slot].as_ref().unwrap()).filter(|edge| {
            let (task, base) = if write {
                (edge.source_task, edge.source_address)
            } else {
                (edge.target_task, edge.target_address)
            };
            task == self.tasks[thread] && address >= base && address < base + edge.bytes as u64
        });
        let edge = matches.next()?;
        assert!(matches.all(|other| write && other.bank == edge.bank
            && other.source_address == edge.source_address && other.bytes == edge.bytes),
            "Ambiguous dataflow storage");
        Some(edge)
    }

    /// Whether `address` is in storage bound to the thread's task, and the address that answer holds up to.
    pub fn span(&self, thread: usize, address: u64, write: bool) -> (bool, u64) {
        if let Some(edge) = self.edge(thread, address, write) {
            let base = if write { edge.source_address } else { edge.target_address };
            return (true, base + edge.bytes as u64);
        }
        let slots = if write { &self.sources[thread] } else { &self.targets[thread] };
        let next = slots
            .iter()
            .map(|&slot| self.edges[slot].as_ref().unwrap())
            .filter(|edge| if write { edge.source_task } else { edge.target_task } == self.tasks[thread])
            .map(|edge| if write { edge.source_address } else { edge.target_address })
            .filter(|&base| base > address)
            .min();
        (false, next.unwrap_or(u64::MAX))
    }

    pub fn contains(&self, thread: usize, address: u64, bytes: usize, write: bool) -> bool {
        if self.edge(thread, address, write).is_none() { return false; }
        let mut consumed = 0;
        while consumed < bytes {
            let at = address + consumed as u64;
            let edge = self.edge(thread, at, write).expect("DMA crosses an unbound dataflow range");
            let base = if write { edge.source_address } else { edge.target_address };
            consumed += (bytes - consumed).min(edge.bytes - (at - base) as usize);
        }
        true
    }

    pub fn transfer_cycles(&self, thread: usize, spans: &[(u64, usize)], write: bool) -> Option<u64> {
        self.edge(thread, spans.first()?.0, write)?;
        let source = self.mesh.thread_nodes[thread];
        let mut cycles = 0;
        let mut previous = None;
        for &(address, bytes) in spans {
            cycles += 2 * (bytes as u64).div_ceil(16);
            let mut consumed = 0;
            while consumed < bytes {
                let at = address + consumed as u64;
                let edge = self.edge(thread, at, write).expect("DMA crosses an unbound dataflow range");
                let base = if write { edge.source_address } else { edge.target_address };
                let offset = (at - base) as usize;
                let bank = edge.bank + offset / self.data[0].len();
                if previous != Some(bank) {
                    let target = bank / self.mesh.banks_per_node;
                    let hops = (source / self.mesh.columns).abs_diff(target / self.mesh.columns)
                        + (source % self.mesh.columns).abs_diff(target % self.mesh.columns);
                    cycles += 24 + 2 * hops as u64;
                    previous = Some(bank);
                }
                consumed += (bytes - consumed).min(edge.bytes - offset)
                    .min(self.data[0].len() - offset % self.data[0].len());
            }
        }
        Some(cycles)
    }


    pub fn control(&mut self, operation: u32, field: u64, value: u64) -> u64 {
        match operation {
            20 => {
                assert!(field < 7);
                self.descriptor[field as usize] = value;
                self.fields |= 1 << field;
                0
            }
            21 => {
                assert_eq!(self.fields, 127, "Incomplete hart dataflow descriptor");
                let [source, source_task, source_address, target, target_task, target_address, bytes] = self.descriptor;
                assert!(source < self.tasks.len() as u64 && target < self.tasks.len() as u64);
                assert!(bytes > 0 && bytes <= (self.data.len() * self.data[0].len()) as u64);
                assert_eq!((source_address | target_address | bytes) % 16, 0);
                source_address.checked_add(bytes).expect("Producer range overflow");
                let end = target_address.checked_add(bytes).expect("Consumer range overflow");
                for edge in self.edges.iter().flatten() {
                    if edge.target == target as usize && edge.target_task == target_task {
                        assert!(
                            end <= edge.target_address || target_address >= edge.target_address + edge.bytes as u64,
                            "Overlapping inputs in one task"
                        );
                    }
                }
                let existing = self.edges.iter().flatten().find(|edge| {
                    edge.source == source as usize
                        && edge.source_task == source_task
                        && edge.source_address == source_address
                        && edge.bytes == bytes as usize
                });
                let (bank, published) = if let Some(edge) = existing {
                    (edge.bank, edge.published)
                } else {
                    let banks = (bytes as usize).div_ceil(self.data[0].len());
                    let bank = (0..=self.data.len() - banks)
                        .find(|&bank| self.edges.iter().flatten().all(|edge| {
                            bank + banks <= edge.bank ||
                                bank >= edge.bank + edge.bytes.div_ceil(self.data[0].len())
                        }))
                        .expect("Hart dataflow storage exhausted");
                    for index in bank..bank + banks {
                        self.initialized[index].fill(false);
                    }
                    (bank, false)
                };
                let slot = self
                    .edges
                    .iter()
                    .position(Option::is_none)
                    .expect("Hart dataflow edges exhausted");
                self.edges[slot] = Some(Edge {
                    source: source as usize,
                    source_task,
                    source_address,
                    target: target as usize,
                    target_task,
                    target_address,
                    bytes: bytes as usize,
                    bank,
                    published,
                });
                self.sources[source as usize].push(slot);
                self.targets[target as usize].push(slot);
                self.fields = 0;
                slot as u64
            }
            22 => {
                let edge = self.edges[field as usize].take().expect("Unknown dataflow edge");
                self.sources[edge.source].retain(|&slot| slot != field as usize);
                self.targets[edge.target].retain(|&slot| slot != field as usize);
                0
            }
            23 => {
                let bank = self.edges[field as usize].as_ref().expect("Unknown producer edge").bank;
                for edge in self.edges.iter_mut().flatten() {
                    if edge.bank == bank {
                        edge.published = true;
                    }
                }
                0
            }
            _ => panic!("Invalid hart dataflow command"),
        }
    }

    pub fn write(&mut self, thread: usize, address: u64, input: &[u8]) -> bool {
        if self.edge(thread, address, true).is_none() { return false; }
        let mut copied = 0;
        while copied < input.len() {
            let at = address + copied as u64;
            let edge = self.edge(thread, at, true)
                .expect("DMA write crosses an unbound dataflow range");
            assert!(!edge.published, "Producer wrote a published value");
            let offset = (at - edge.source_address) as usize;
            let bank = edge.bank + offset / self.data[0].len();
            let row = offset % self.data[0].len();
            let bytes = (input.len() - copied).min(edge.bytes - offset).min(self.data[bank].len() - row);
            self.data[bank][row..row + bytes].copy_from_slice(&input[copied..copied + bytes]);
            self.initialized[bank][row..row + bytes].fill(true);
            copied += bytes;
        }
        true
    }

    pub fn read(&self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        if self.edge(thread, address, false).is_none() { return false; }
        let mut copied = 0;
        while copied < output.len() {
            let at = address + copied as u64;
            let edge = self.edge(thread, at, false)
                .expect("DMA read crosses an unbound dataflow range");
            assert!(edge.published, "Unpublished dataflow input");
            let offset = (at - edge.target_address) as usize;
            let bank = edge.bank + offset / self.data[0].len();
            let row = offset % self.data[0].len();
            let bytes = (output.len() - copied).min(edge.bytes - offset).min(self.data[bank].len() - row);
            assert!(self.initialized[bank][row..row + bytes].iter().all(|&valid| valid),
                "Producer did not initialize the consumed bytes");
            output[copied..copied + bytes].copy_from_slice(&self.data[bank][row..row + bytes]);
            copied += bytes;
        }
        true
    }
}
