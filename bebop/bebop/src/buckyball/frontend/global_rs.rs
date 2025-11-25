/// Global Reservation Station - manages instruction issue and completion across domains
use crate::buckyball::builtin::{Module, Wire};
use crate::buckyball::frontend::DecoderOutput;

/// ROB (Reorder Buffer) ID type
pub type RobId = usize;

/// Issue command from Global RS to domain
#[derive(Clone, Default)]
pub struct GlobalRsIssue {
    pub rob_id: RobId,
    pub cmd: DecoderOutput,
}

/// Completion signal from domain to Global RS
#[derive(Clone, Default)]
pub struct GlobalRsComplete {
    pub rob_id: RobId,
    pub data: u32, // Result data (if any)
}

/// ROB Entry
#[derive(Clone)]
struct RobEntry {
    valid: bool,
    issued: bool,
    completed: bool,
    rob_id: RobId,
    is_mem: bool,
    data: u32,
}

impl Default for RobEntry {
    fn default() -> Self {
        Self {
            valid: false,
            issued: false,
            completed: false,
            rob_id: 0,
            is_mem: false,
            data: 0,
        }
    }
}

/// Global Reservation Station
/// Manages instruction issue to Memory and Ball domains and tracks completion via ROB
pub struct GlobalReservationStation {
    name: String,

    // Input from GlobalDecoder
    pub input: Wire<DecoderOutput>,

    // Output to Memory Domain
    pub mem_issue: Wire<GlobalRsIssue>,

    // Output to Ball Domain
    pub ball_issue: Wire<GlobalRsIssue>,

    // Input from Memory Domain (completion)
    pub mem_complete: Wire<GlobalRsComplete>,

    // Input from Ball Domain (completion)
    pub ball_complete: Wire<GlobalRsComplete>,

    // ROB (Reorder Buffer)
    rob: Vec<RobEntry>,
    rob_size: usize,
    head_ptr: usize,
    tail_ptr: usize,

    // Response data (for external query)
    last_response: u32,
}

impl GlobalReservationStation {
    pub fn new(name: impl Into<String>, rob_size: usize) -> Self {
        Self {
            name: name.into(),
            input: Wire::default(),
            mem_issue: Wire::default(),
            ball_issue: Wire::default(),
            mem_complete: Wire::default(),
            ball_complete: Wire::default(),
            rob: vec![RobEntry::default(); rob_size],
            rob_size,
            head_ptr: 0,
            tail_ptr: 0,
            last_response: 0,
        }
    }

    /// Check if ROB is full
    fn is_rob_full(&self) -> bool {
        let next_tail = (self.tail_ptr + 1) % self.rob_size;
        next_tail == self.head_ptr && self.rob[self.head_ptr].valid
    }

    /// Check if ROB is empty
    fn is_rob_empty(&self) -> bool {
        self.head_ptr == self.tail_ptr && !self.rob[self.head_ptr].valid
    }

    /// Allocate ROB entry
    fn allocate_rob(&mut self) -> Option<RobId> {
        if self.is_rob_full() {
            return None;
        }

        let rob_id = self.tail_ptr;
        self.rob[rob_id].valid = true;
        self.rob[rob_id].issued = false;
        self.rob[rob_id].completed = false;
        self.rob[rob_id].rob_id = rob_id;

        self.tail_ptr = (self.tail_ptr + 1) % self.rob_size;
        Some(rob_id)
    }

    /// Commit ROB entry (remove from head)
    fn commit_rob(&mut self) {
        if !self.is_rob_empty() && self.rob[self.head_ptr].completed {
            self.last_response = self.rob[self.head_ptr].data;
            self.rob[self.head_ptr].valid = false;
            self.head_ptr = (self.head_ptr + 1) % self.rob_size;
        }
    }

    /// Get last response data
    pub fn get_response(&self) -> u32 {
        self.last_response
    }
}

impl Module for GlobalReservationStation {
    fn run(&mut self) {
        // 1. Handle completion from Memory Domain
        if self.mem_complete.valid {
            let rob_id = self.mem_complete.value.rob_id;
            if rob_id < self.rob_size && self.rob[rob_id].valid {
                self.rob[rob_id].completed = true;
                self.rob[rob_id].data = self.mem_complete.value.data;
            }
        }

        // Handle completion from Ball Domain
        if self.ball_complete.valid {
            let rob_id = self.ball_complete.value.rob_id;
            if rob_id < self.rob_size && self.rob[rob_id].valid {
                self.rob[rob_id].completed = true;
                self.rob[rob_id].data = self.ball_complete.value.data;
            }
        }

        // 2. Commit completed instructions from ROB head
        self.commit_rob();

        // 3. Issue new instruction if input is valid and ROB has space
        if self.input.valid && !self.is_rob_full() {
            if let Some(rob_id) = self.allocate_rob() {
                let funct = self.input.value.funct;

                // Determine if it's memory (24-25) or ball (32-42) instruction
                let is_mem = funct == 24 || funct == 25;
                let is_ball = funct >= 32 && funct <= 42;

                self.rob[rob_id].is_mem = is_mem;
                self.rob[rob_id].issued = true;

                if is_mem {
                    // Issue to Memory Domain
                    self.mem_issue.set(GlobalRsIssue {
                        rob_id,
                        cmd: self.input.value.clone(),
                    });
                    self.ball_issue.clear();
                } else if is_ball {
                    // Issue to Ball Domain
                    self.ball_issue.set(GlobalRsIssue {
                        rob_id,
                        cmd: self.input.value.clone(),
                    });
                    self.mem_issue.clear();
                } else {
                    self.mem_issue.clear();
                    self.ball_issue.clear();
                }
            }
        } else {
            self.mem_issue.clear();
            self.ball_issue.clear();
        }
    }

    fn reset(&mut self) {
        self.input = Wire::default();
        self.mem_issue = Wire::default();
        self.ball_issue = Wire::default();
        self.mem_complete = Wire::default();
        self.ball_complete = Wire::default();
        self.rob = vec![RobEntry::default(); self.rob_size];
        self.head_ptr = 0;
        self.tail_ptr = 0;
        self.last_response = 0;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
