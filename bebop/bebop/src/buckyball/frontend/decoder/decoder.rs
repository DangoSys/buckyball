/// GlobalDecoder - 全局指令解码器
/// 只负责识别指令类型，分发到对应 domain

/// 指令目标域
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDomain {
  MemDomain,
  BallDomain,
  Unknown,
}

/// 解码后的指令
#[derive(Debug, Clone)]
pub struct DecodedInstr {
  pub target: TargetDomain,
  pub funct: u64,
  pub xs1: u64,
  pub xs2: u64,
}

/// 全局解码器
pub struct GlobalDecoder {
  name: String,
}

impl GlobalDecoder {
  pub fn new(name: impl Into<String>) -> Self {
    Self { name: name.into() }
  }

  /// 解码指令，确定目标域
  pub fn decode(&self, funct: u64, xs1: u64, xs2: u64) -> DecodedInstr {
    let target = match funct {
      24 | 25 => TargetDomain::MemDomain,      // MVIN, MVOUT
      32..=42 => TargetDomain::BallDomain,     // Ball instructions
      _ => TargetDomain::Unknown,
    };

    DecodedInstr { target, funct, xs1, xs2 }
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &str {
    &self.name
  }
}
