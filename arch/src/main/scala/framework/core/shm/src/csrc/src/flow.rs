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

/// Core-local data edges. This model has no device communication state.
pub struct Mesh {
    pub columns: usize,
    pub thread_nodes: Vec<usize>,
    pub banks_per_node: usize,
}

pub struct Storage {
    mesh: Mesh,
    tasks: Vec<u64>,
    edges: Vec<Option<Edge>>,
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
            data: vec![vec![0; bytes_per_bank]; banks],
            initialized: vec![vec![false; bytes_per_bank]; banks],
            descriptor: [0; 7],
            fields: 0,
        }
    }

    pub fn start(&mut self, thread: usize, task: u64) {
        self.tasks[thread] = task;
    }

    pub fn transfer_cycles(
        &self,
        thread: usize,
        address: u64,
        rows: u64,
        write: bool,
    ) -> Option<u64> {
        let edge = self.edges.iter().flatten().find(|edge| {
            let (owner, task, base) = if write {
                (edge.source, edge.source_task, edge.source_address)
            } else {
                (edge.target, edge.target_task, edge.target_address)
            };
            owner == thread
                && task == self.tasks[thread]
                && address >= base
                && address < base + edge.bytes as u64
        })?;
        let source = self.mesh.thread_nodes[thread];
        let target = edge.bank / self.mesh.banks_per_node;
        let hops = (source / self.mesh.columns).abs_diff(target / self.mesh.columns)
            + (source % self.mesh.columns).abs_diff(target % self.mesh.columns);
        Some(rows * if write { 6 } else { 5 } + 2 * hops as u64)
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
                assert_eq!(self.fields, 127, "Incomplete core dataflow descriptor");
                let [source, source_task, source_address, target, target_task, target_address, bytes] =
                    self.descriptor;
                assert!(source < self.tasks.len() as u64 && target < self.tasks.len() as u64);
                assert!(bytes > 0 && bytes <= self.data[0].len() as u64);
                assert_eq!((source_address | target_address | bytes) % 16, 0);
                source_address
                    .checked_add(bytes)
                    .expect("Producer range overflow");
                let end = target_address
                    .checked_add(bytes)
                    .expect("Consumer range overflow");
                for edge in self.edges.iter().flatten() {
                    if edge.target == target as usize && edge.target_task == target_task {
                        assert!(
                            end <= edge.target_address
                                || target_address >= edge.target_address + edge.bytes as u64,
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
                    let bank = (0..self.data.len())
                        .find(|&bank| self.edges.iter().flatten().all(|edge| edge.bank != bank))
                        .expect("Core dataflow storage exhausted");
                    self.initialized[bank].fill(false);
                    (bank, false)
                };
                let slot = self
                    .edges
                    .iter()
                    .position(Option::is_none)
                    .expect("Core dataflow edges exhausted");
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
                self.fields = 0;
                slot as u64
            }
            22 => {
                self.edges[field as usize]
                    .take()
                    .expect("Unknown dataflow edge");
                0
            }
            23 => {
                let bank = self.edges[field as usize]
                    .as_ref()
                    .expect("Unknown producer edge")
                    .bank;
                for edge in self.edges.iter_mut().flatten() {
                    if edge.bank == bank {
                        edge.published = true;
                    }
                }
                0
            }
            _ => panic!("Invalid core dataflow command"),
        }
    }

    pub fn write(&mut self, thread: usize, address: u64, input: &[u8]) -> bool {
        let matches = self
            .edges
            .iter()
            .flatten()
            .filter(|edge| {
                edge.source == thread
                    && edge.source_task == self.tasks[thread]
                    && address >= edge.source_address
                    && address < edge.source_address + edge.bytes as u64
            })
            .collect::<Vec<_>>();
        let Some(edge) = matches.first() else {
            return false;
        };
        assert!(
            matches.iter().all(|other| other.bank == edge.bank),
            "Ambiguous producer storage"
        );
        assert!(!edge.published, "Producer wrote a published value");
        let offset = (address - edge.source_address) as usize;
        assert!(
            offset + input.len() <= edge.bytes,
            "Write crosses a dataflow boundary"
        );
        self.data[edge.bank][offset..offset + input.len()].copy_from_slice(input);
        self.initialized[edge.bank][offset..offset + input.len()].fill(true);
        true
    }

    pub fn read(&mut self, thread: usize, address: u64, output: &mut [u8]) -> bool {
        let mut matches = self.edges.iter().flatten().filter(|edge| {
            edge.target == thread
                && edge.target_task == self.tasks[thread]
                && address >= edge.target_address
                && address < edge.target_address + edge.bytes as u64
        });
        let Some(edge) = matches.next() else {
            return false;
        };
        assert!(
            matches.next().is_none() && edge.published,
            "Ambiguous or unpublished input"
        );
        let offset = (address - edge.target_address) as usize;
        assert!(
            offset + output.len() <= edge.bytes,
            "Read crosses a dataflow boundary"
        );
        assert!(
            self.initialized[edge.bank][offset..offset + output.len()]
                .iter()
                .all(|&valid| valid),
            "Producer did not initialize the consumed bytes"
        );
        output.copy_from_slice(&self.data[edge.bank][offset..offset + output.len()]);
        true
    }
}
