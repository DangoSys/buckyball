use super::decode;
use super::instruction::ExecContext;

const FUNCT_MVOUT: u32 = 16;
const FUNCT_MSET: u32 = 32;
const FUNCT_MVIN: u32 = 33;
const FUNCT_PRELOAD: u32 = 53;
const FUNCT_COMPUTE_PRELOADED: u32 = 66;
const FUNCT_COMPUTE_ACCUMULATED: u32 = 67;
const MAX_DMA_STRIDE: u64 = (1 << 19) - 1;

fn execute(funct: u32, xs1: u64, xs2: u64, ctx: &mut ExecContext) {
    decode::execute_known(funct, xs1, xs2, ctx)
        .unwrap_or_else(|| panic!("Gemmini loop emitted unsupported micro-op funct7={funct}"));
}

fn banks(bank0: u64, bank1: u64, bank2: u64, iter: u64) -> u64 {
    assert!(bank0 < (1 << 10), "Gemmini loop bank0 exceeds 10 bits");
    assert!(bank1 < (1 << 10), "Gemmini loop bank1 exceeds 10 bits");
    assert!(bank2 < (1 << 10), "Gemmini loop bank2 exceeds 10 bits");
    assert!(iter < (1 << 34), "Gemmini loop iter exceeds 34 bits");
    bank0 | (bank1 << 10) | (bank2 << 20) | (iter << 30)
}

fn dma(addr: u64, stride: u64) -> u64 {
    assert!(
        addr < (1 << 39),
        "Gemmini loop DMA address exceeds 39 bits: 0x{addr:x}"
    );
    assert!(
        stride > 0 && stride <= MAX_DMA_STRIDE,
        "Gemmini loop DMA stride out of range: {stride}"
    );
    addr | (stride << 39)
}

pub fn checked_stride(bytes: u64, row_bytes: u64, name: &str) -> u64 {
    assert!(bytes > 0, "{name} must be > 0");
    assert!(row_bytes > 0, "{name}: row width must be > 0");
    assert_eq!(
        bytes % row_bytes,
        0,
        "{name} must be divisible by {row_bytes} bytes"
    );
    let stride = bytes / row_bytes;
    assert!(stride <= MAX_DMA_STRIDE, "{name} exceeds 19-bit DMA stride");
    stride
}

pub fn alloc(ctx: &mut ExecContext, bank: u64, groups: u64) {
    assert!(
        groups > 0 && groups < (1 << 5),
        "Gemmini loop invalid bank groups: {groups}"
    );
    let xs2 = 1 | (groups << 5) | (1 << 10);
    execute(FUNCT_MSET, banks(bank, 0, 0, 0), xs2, ctx);
}

pub fn mvin(ctx: &mut ExecContext, bank: u64, addr: u64, iter: u64, stride: u64) {
    execute(FUNCT_MVIN, banks(bank, 0, 0, iter), dma(addr, stride), ctx);
}

pub fn mvout(ctx: &mut ExecContext, bank: u64, addr: u64, iter: u64, stride: u64) {
    execute(FUNCT_MVOUT, banks(bank, 0, 0, iter), dma(addr, stride), ctx);
}

pub fn preload(ctx: &mut ExecContext, source: u64, output: u64, iter: u64) {
    execute(FUNCT_PRELOAD, banks(source, 0, output, iter), 1, ctx);
}

pub fn compute(
    ctx: &mut ExecContext,
    accumulated: bool,
    op1: u64,
    op2: u64,
    output: u64,
    iter: u64,
    zero_op2: bool,
    zero_op1_tail: bool,
) {
    let funct = if accumulated {
        FUNCT_COMPUTE_ACCUMULATED
    } else {
        FUNCT_COMPUTE_PRELOADED
    };
    let mode = if accumulated { 3 } else { 2 };
    let xs2 = mode | ((zero_op2 as u64) << 4) | ((zero_op1_tail as u64) << 5);
    execute(funct, banks(op1, op2, output, iter), xs2, ctx);
}

pub fn free_after_digest(ctx: &mut ExecContext, bank: u64) {
    ctx.defer_bank_free(bank);
}
