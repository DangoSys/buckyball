//===- im2colOps.td - Im2colBall operations ----------------*- tablegen -*-===//
//
// im2col_i8: the input bank holds a square feature map of `input_side` pixels
// per side; group g is channel block g, line y * input_side + x holds the
// sixteen channels of pixel (y, x). The output receives the windows of the
// `kernel` x `kernel` convolution with `stride` and zero `padding`, every
// window in row-major order, as the matmul_i8 A operand: K is ordered (kernel
// row, kernel column, channel), sixteen columns (one input line) per K tile,
// K tile t = (kr * kernel + kc) * groups + g. The output groups divide the K
// tiles; group j holds the next K / 16 / groups tiles in M tiles of
// `tile_rows` rows (the layout of the matmul that consumes it), windows beyond
// the last zero up to a multiple of tile_rows.
//
//===----------------------------------------------------------------------===//

#ifndef IM2COL_OPS
#define IM2COL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def Im2colBall_Dialect : Dialect {
  let name = "im2col_ball";
  let cppNamespace = "::buddy::im2col_ball";
  let summary = "The operations of Im2colBall";
}

class Im2colBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Im2colBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankIm2colI8Op : Im2colBall_Op<"bank_im2col_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "im2col_i8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$input_side,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding,
                       I64Attr:$tile_rows);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def Im2colI8Op : Im2colBall_Op<"im2col_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "im2col_i8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$input_side,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding,
                       I64Attr:$tile_rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // IM2COL_OPS
