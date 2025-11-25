/// BallDomain Decoder - decodes ball instructions
use crate::buckyball::builtin::{Module, Wire};
use crate::buckyball::frontend::DecoderOutput;

#[derive(Clone, Default)]
pub struct BallDecodeCmd {
    pub bid: u8,
    pub xs1: u64,
    pub xs2: u64,
}

pub struct DomainDecoder {
    name: String,
    pub input: Wire<DecoderOutput>,
    pub output: Wire<BallDecodeCmd>,
}

impl DomainDecoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: Wire::default(),
            output: Wire::default(),
        }
    }
}

impl Module for DomainDecoder {
    fn run(&mut self) {
        if !self.input.valid {
            self.output.clear();
            return;
        }

        let input = &self.input.value;

        // Map funct to bid
        let bid = match input.funct {
            32 => 0, // VEC
            33 => 2, // IM2COL
            34 => 3, // TRANSPOSE
            38 => 4, // RELU
            42 => 1, // ABFT_SYSTOLIC (MatrixBall)
            _ => {
                self.output.clear();
                return;
            }
        };

        self.output.set(BallDecodeCmd {
            bid,
            xs1: input.xs1,
            xs2: input.xs2,
        });
    }

    fn reset(&mut self) {
        self.input = Wire::default();
        self.output = Wire::default();
    }

    fn name(&self) -> &str {
        &self.name
    }
}
