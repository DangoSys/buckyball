/// BBus - Ball bus with registration pattern
use crate::builtin::{Module, Wire};
use crate::buckyball::frontend::GlobalRsComplete;
use crate::buckyball::prototype::*;
use super::rs::BallIssue;

const NUM_BALLS: usize = 5;

/// BBus manages all registered balls
pub struct BBus {
    name: String,

    // Registered balls (matching RTL busRegister.scala)
    vec_ball: VecBall,      // bid 0
    matrix_ball: MatrixBall, // bid 1
    im2col_ball: Im2colBall, // bid 2
    transpose_ball: TransposeBall, // bid 3
    relu_ball: ReluBall,    // bid 4

    pub cmd_reqs: [Wire<BallIssue>; NUM_BALLS],
    pub cmd_resps: [Wire<GlobalRsComplete>; NUM_BALLS],
}

impl BBus {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            // Register balls with their bid
            vec_ball: VecBall::new(0),
            matrix_ball: MatrixBall::new(1),
            im2col_ball: Im2colBall::new(2),
            transpose_ball: TransposeBall::new(3),
            relu_ball: ReluBall::new(4),
            cmd_reqs: Default::default(),
            cmd_resps: Default::default(),
        }
    }
}

impl Module for BBus {
    fn run(&mut self) {
        // Run all registered balls
        self.vec_ball.run();
        self.matrix_ball.run();
        self.im2col_ball.run();
        self.transpose_ball.run();
        self.relu_ball.run();

        // Route commands to balls based on bid
        // Each ball only sees its own instruction format (ISA isolation)
        use crate::buckyball::prototype::vector::VecCmd;
        use crate::buckyball::prototype::matrix::MatrixCmd;
        use crate::buckyball::prototype::im2col::Im2colCmd;
        use crate::buckyball::prototype::transpose::TransposeCmd;
        use crate::buckyball::prototype::relu::ReluCmd;

        if self.cmd_reqs[0].valid {
            let issue = &self.cmd_reqs[0].value;
            let cmd = VecCmd::from_fields(issue.xs1, issue.xs2);
            self.vec_ball.cmd_req.set(cmd);
        }

        if self.cmd_reqs[1].valid {
            let issue = &self.cmd_reqs[1].value;
            let cmd = MatrixCmd::from_fields(issue.xs1, issue.xs2);
            self.matrix_ball.cmd_req.set(cmd);
        }

        if self.cmd_reqs[2].valid {
            let issue = &self.cmd_reqs[2].value;
            let cmd = Im2colCmd::from_fields(issue.xs1, issue.xs2);
            self.im2col_ball.cmd_req.set(cmd);
        }

        if self.cmd_reqs[3].valid {
            let issue = &self.cmd_reqs[3].value;
            let cmd = TransposeCmd::from_fields(issue.xs1, issue.xs2);
            self.transpose_ball.cmd_req.set(cmd);
        }

        if self.cmd_reqs[4].valid {
            let issue = &self.cmd_reqs[4].value;
            let cmd = ReluCmd::from_fields(issue.xs1, issue.xs2);
            self.relu_ball.cmd_req.set(cmd);
        }

        // Collect responses
        for i in 0..NUM_BALLS {
            self.cmd_resps[i].clear();
        }

        if self.vec_ball.cmd_resp.valid && self.cmd_reqs[0].valid {
            self.cmd_resps[0].set(GlobalRsComplete {
                rob_id: self.cmd_reqs[0].value.rob_id,
                data: self.vec_ball.cmd_resp.value
            });
        }

        if self.matrix_ball.cmd_resp.valid && self.cmd_reqs[1].valid {
            self.cmd_resps[1].set(GlobalRsComplete {
                rob_id: self.cmd_reqs[1].value.rob_id,
                data: self.matrix_ball.cmd_resp.value
            });
        }

        if self.im2col_ball.cmd_resp.valid && self.cmd_reqs[2].valid {
            self.cmd_resps[2].set(GlobalRsComplete {
                rob_id: self.cmd_reqs[2].value.rob_id,
                data: self.im2col_ball.cmd_resp.value
            });
        }

        if self.transpose_ball.cmd_resp.valid && self.cmd_reqs[3].valid {
            self.cmd_resps[3].set(GlobalRsComplete {
                rob_id: self.cmd_reqs[3].value.rob_id,
                data: self.transpose_ball.cmd_resp.value
            });
        }

        if self.relu_ball.cmd_resp.valid && self.cmd_reqs[4].valid {
            self.cmd_resps[4].set(GlobalRsComplete {
                rob_id: self.cmd_reqs[4].value.rob_id,
                data: self.relu_ball.cmd_resp.value
            });
        }
    }

    fn reset(&mut self) {
        self.vec_ball.reset();
        self.matrix_ball.reset();
        self.im2col_ball.reset();
        self.transpose_ball.reset();
        self.relu_ball.reset();
        self.cmd_reqs = Default::default();
        self.cmd_resps = Default::default();
    }

    fn name(&self) -> &str {
        &self.name
    }
}
