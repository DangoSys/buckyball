/// BallDomain Reservation Station
use crate::builtin::{Module, Wire};
use crate::buckyball::frontend::{GlobalRsComplete, GlobalRsIssue, RobId};

const NUM_BALLS: usize = 5;

#[derive(Clone, Default)]
pub struct BallIssue {
    pub rob_id: RobId,
    pub bid: u8,
    pub xs1: u64,
    pub xs2: u64,
}

pub struct ReservationStation {
    name: String,

    pub decode_input: Wire<GlobalRsIssue>,
    pub ball_issues: [Wire<BallIssue>; NUM_BALLS],
    pub ball_completes: [Wire<GlobalRsComplete>; NUM_BALLS],
    pub complete_output: Wire<GlobalRsComplete>,
}

impl ReservationStation {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            decode_input: Wire::default(),
            ball_issues: Default::default(),
            ball_completes: Default::default(),
            complete_output: Wire::default(),
        }
    }
}

impl Module for ReservationStation {
    fn run(&mut self) {
        // Forward completion from any ball
        self.complete_output.clear();
        for i in 0..NUM_BALLS {
            if self.ball_completes[i].valid {
                self.complete_output.set(self.ball_completes[i].value.clone());
                break;
            }
        }

        // Clear all issues
        for i in 0..NUM_BALLS {
            self.ball_issues[i].clear();
        }

        // Issue to appropriate ball
        if self.decode_input.valid {
            let cmd = &self.decode_input.value.cmd;
            let rob_id = self.decode_input.value.rob_id;

            let bid = match cmd.funct {
                32 => 0, // VEC
                33 => 2, // IM2COL
                34 => 3, // TRANSPOSE
                38 => 4, // RELU
                42 => 1, // MATRIX
                _ => return,
            };

            if (bid as usize) < NUM_BALLS {
                self.ball_issues[bid as usize].set(BallIssue {
                    rob_id,
                    bid,
                    xs1: cmd.xs1,
                    xs2: cmd.xs2,
                });
            }
        }
    }

    fn reset(&mut self) {
        self.decode_input = Wire::default();
        self.ball_issues = Default::default();
        self.ball_completes = Default::default();
        self.complete_output = Wire::default();
    }

    fn name(&self) -> &str {
        &self.name
    }
}
