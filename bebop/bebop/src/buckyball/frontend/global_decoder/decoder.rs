/// Global Decoder Module - decoder module
use crate::builtin::{Module, Wire};

/// Global Decoder input
#[derive(Clone, Default)]
pub struct DecoderInput {
  pub funct: u64,
  pub xs1: u64,
  pub xs2: u64,
}

/// Global Decoder output
#[derive(Clone, Default)]
pub struct DecoderOutput {
  pub funct: u64, // Original instruction code
  pub xs1: u64,
  pub xs2: u64,
}

/// Global Decoder - global decoder
pub struct Decoder {
  name: String,

  // Input
  pub input: Wire<DecoderInput>,

  // Output
  pub output: Wire<DecoderOutput>,
}

impl Decoder {
  pub fn new(name: impl Into<String>) -> Self {
    Self {
      name: name.into(),
      input: Wire::default(),
      output: Wire::default(),
    }
  }
}

impl Module for Decoder {
  fn run(&mut self) {
    if !self.input.valid {
      self.output.clear();
      return;
    }

    let input = &self.input.value;

    // Decode logic: only responsible for routing, pass original instruction
    let output = DecoderOutput {
      funct: input.funct,
      xs1: input.xs1,
      xs2: input.xs2,
    };

    // Route instruction based on funct7 value
    // MVIN_FUNC7 = 24 (0x18)
    // MVOUT_FUNC7 = 25 (0x19)
    let valid = match input.funct {
      24 => {
        // MVIN - route to memory domain
        println!(
          "[Decoder] MVIN route to memory domain: funct={}, xs1=0x{:x}, xs2=0x{:x}",
          input.funct, input.xs1, input.xs2
        );
        true
      },
      25 => {
        // MVOUT - route to memory domain
        println!(
          "[Decoder] MVOUT route to memory domain: funct={}, xs1=0x{:x}, xs2=0x{:x}",
          input.funct, input.xs1, input.xs2
        );
        true
      },
      _ => {
        println!(
          "[Decoder] UNKNOWN: funct={}, xs1=0x{:x}, xs2=0x{:x}",
          input.funct, input.xs1, input.xs2
        );
        false
      },
    };

    if valid {
      self.output.set(output);
    } else {
      self.output.clear();
    }
  }

  fn reset(&mut self) {
    self.input = Wire::default();
    self.output = Wire::default();
  }

  fn name(&self) -> &str {
    &self.name
  }
}
