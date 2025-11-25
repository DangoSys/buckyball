/// MemDomain Reservation Station
use crate::buckyball::builtin::{Module, Wire};
use crate::buckyball::frontend::{GlobalRsComplete, GlobalRsIssue, MvinConfig, MvoutConfig};
use super::{MemLoaderReq, MemStorerReq};

#[derive(Clone, Default)]
pub struct IssueOutput {
    pub ld: Wire<MemLoaderReq>,
    pub st: Wire<MemStorerReq>,
}

#[derive(Clone, Default)]
pub struct CommitInput {
    pub ld: Wire<GlobalRsComplete>,
    pub st: Wire<GlobalRsComplete>,
}

pub struct ReservationStation {
    name: String,

    pub decode_input: Wire<GlobalRsIssue>,
    pub issue_output: IssueOutput,
    pub commit_input: CommitInput,
    pub complete_output: Wire<GlobalRsComplete>,
}

impl ReservationStation {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            decode_input: Wire::default(),
            issue_output: IssueOutput::default(),
            commit_input: CommitInput::default(),
            complete_output: Wire::default(),
        }
    }
}

impl Module for ReservationStation {
    fn run(&mut self) {
        // Forward completion from MemLoader/MemStorer to Global RS
        if self.commit_input.ld.valid {
            self.complete_output.set(self.commit_input.ld.value.clone());
        } else if self.commit_input.st.valid {
            self.complete_output.set(self.commit_input.st.value.clone());
        } else {
            self.complete_output.clear();
        }

        // Issue to MemLoader or MemStorer based on decoded command
        if self.decode_input.valid {
            let cmd = &self.decode_input.value.cmd;
            let rob_id = self.decode_input.value.rob_id;

            match cmd.funct {
                24 => {
                    // MVIN - issue to MemLoader
                    let config = MvinConfig::from_fields(cmd.xs1, cmd.xs2);
                    self.issue_output.ld.set(MemLoaderReq { rob_id, config });
                    self.issue_output.st.clear();
                },
                25 => {
                    // MVOUT - issue to MemStorer
                    let config = MvoutConfig::from_fields(cmd.xs1, cmd.xs2);
                    self.issue_output.st.set(MemStorerReq { rob_id, config });
                    self.issue_output.ld.clear();
                },
                _ => {
                    self.issue_output.ld.clear();
                    self.issue_output.st.clear();
                }
            }
        } else {
            self.issue_output.ld.clear();
            self.issue_output.st.clear();
        }
    }

    fn reset(&mut self) {
        self.decode_input = Wire::default();
        self.issue_output = IssueOutput::default();
        self.commit_input = CommitInput::default();
        self.complete_output = Wire::default();
    }

    fn name(&self) -> &str {
        &self.name
    }
}
