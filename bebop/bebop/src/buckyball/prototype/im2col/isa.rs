/// Im2col Ball ISA
#[derive(Clone, Default, Debug)]
pub struct Im2colCmd {
    pub op1_addr: u32,
    pub op2_addr: u32,
    pub dst_addr: u32,
    pub iter: u32,
}

impl Im2colCmd {
    pub fn from_fields(xs1: u64, xs2: u64) -> Self {
        Self {
            op1_addr: (xs1 & 0x3FFF) as u32,
            op2_addr: ((xs1 >> 14) & 0x3FFF) as u32,
            dst_addr: (xs2 & 0x3FFF) as u32,
            iter: ((xs2 >> 14) & 0x3FF) as u32,
        }
    }
}
