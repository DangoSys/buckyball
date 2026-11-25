use std::collections::VecDeque;

/// The reservation-station queues of Gemmini, each with its controller.
pub const LOAD: usize = 0;
pub const EXECUTE: usize = 1;
pub const STORE: usize = 2;

/// Entries of the reservation station per queue (DefaultGemmini).
const ENTRIES: [usize; 3] = [8, 16, 4];

/// The scratchpad's rows, then the accumulator's.
const SPAD_ROWS: usize = 4 * 4096;
const ACC_ROWS: usize = 1024;

struct Queue {
    /// When the controller can start its next instruction.
    free: u64,
    /// Completion of the instructions holding an entry.
    entries: VecDeque<u64>,
}

/// When an instruction last wrote or read a row, and on which queue.
#[derive(Clone, Copy)]
struct Access {
    cycle: u64,
    queue: usize,
}

/// The timing of Gemmini: instructions are presented in order to the
/// reservation station, each queue's controller runs its instructions in
/// order, and an instruction waits for the rows it reads until they are
/// written, and for the rows it writes until an instruction of another queue
/// has written or read them.
pub struct Timing {
    /// When the frontend presents the next instruction.
    pub cursor: u64,
    /// When the last instruction of the CPU was accepted.
    pub accepted: u64,
    /// Completion of all submitted work.
    pub complete: u64,
    /// The busy cycles before the current stretch of work, and its start.
    busy_before: u64,
    busy_start: u64,
    queues: [Queue; 3],
    /// Completion of the stores issued so far: a load follows them, as it may
    /// read what they write.
    stored: u64,
    written: Vec<Access>,
    read: Vec<Access>,
    reads: Vec<usize>,
    writes: Vec<usize>,
}

impl Timing {
    pub fn new() -> Self {
        let idle = Access { cycle: 0, queue: LOAD };
        Timing {
            cursor: 0,
            accepted: 0,
            complete: 0,
            busy_before: 0,
            busy_start: 0,
            queues: std::array::from_fn(|_| Queue {
                free: 0,
                entries: VecDeque::new(),
            }),
            stored: 0,
            written: vec![idle; SPAD_ROWS + ACC_ROWS],
            read: vec![idle; SPAD_ROWS + ACC_ROWS],
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    fn row(accumulator: bool, row: usize) -> usize {
        if accumulator {
            SPAD_ROWS + row
        } else {
            row
        }
    }

    /// The next instruction reads `row`.
    pub fn read(&mut self, accumulator: bool, row: usize) {
        self.reads.push(Self::row(accumulator, row));
    }

    /// The next instruction writes `row`.
    pub fn write(&mut self, accumulator: bool, row: usize) {
        self.writes.push(Self::row(accumulator, row));
    }

    /// The cycles Gemmini was busy before `cycle`.
    pub fn busy(&self, cycle: u64) -> u64 {
        self.busy_before + cycle.min(self.complete).saturating_sub(self.busy_start)
    }

    /// Instructions of `queue` still holding an entry of the reservation station at `cycle`.
    pub fn pending(&self, queue: usize, cycle: u64) -> usize {
        self.queues[queue].entries.iter().filter(|done| **done > cycle).count()
    }

    /// Issues an instruction to `queue` with the footprint given since the last one: it takes
    /// an entry of the reservation station, then occupies the controller for `occupancy`
    /// cycles; its results are ready `latency` cycles later.
    pub fn issue(&mut self, queue: usize, occupancy: u64, latency: u64) {
        let entries = &mut self.queues[queue].entries;
        let cursor = self.cursor;
        entries.retain(|done| *done > cursor);
        if entries.len() == ENTRIES[queue] {
            // The first entry to complete makes room.
            let (index, done) = entries.iter().enumerate().min_by_key(|(_, done)| **done).unwrap();
            self.cursor = self.cursor.max(*done);
            entries.remove(index);
        }
        self.accepted = self.cursor;
        if self.cursor >= self.complete {
            self.busy_before += self.complete - self.busy_start;
            self.busy_start = self.cursor;
        }
        let mut start = self.cursor.max(self.queues[queue].free);
        if queue == LOAD {
            start = start.max(self.stored);
        }
        for row in &self.reads {
            start = start.max(self.written[*row].cycle);
        }
        for row in &self.writes {
            for access in [self.written[*row], self.read[*row]] {
                if access.queue != queue {
                    start = start.max(access.cycle);
                }
            }
        }
        let busy = start + occupancy;
        let ready = busy + latency;
        for row in self.reads.drain(..) {
            self.read[row] = Access {
                cycle: self.read[row].cycle.max(busy),
                queue,
            };
        }
        for row in self.writes.drain(..) {
            self.written[row] = Access { cycle: ready, queue };
        }
        self.queues[queue].free = busy;
        if queue == STORE {
            self.stored = self.stored.max(ready);
        }
        // A store leaves the reservation station once its writes are sent.
        self.queues[queue]
            .entries
            .push_back(if queue == STORE { busy } else { ready });
        self.complete = self.complete.max(ready);
    }
}
