#ifndef F32ADDBALL_OPS
#define F32ADDBALL_OPS

include "Buckyball.td"

def F32AddOp : Buckyball_Op<"f32add", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Ordered RNE accumulator plus a selected FP32 source group";
  let description = [{
    Each 128-bit row contains four FP32 elements. For each element, compute
    accumulator + source with RNE and canonical NaN 0x7fc00000. When first is
    true, use positive zero without reading b. The three virtual banks must
    differ. Only a uses the selected group; b and c each use group zero.
  }];
  let arguments = (ins I64:$a, I64:$b, I64:$c,
                       I64Attr:$rows, I64Attr:$group, BoolAttr:$first);
  let hasVerifier = 1;
  let assemblyFormat = "$a `,` $b `,` $c prop-dict attr-dict `:` type($a)";
}

def BankF32AddOp : Buckyball_Op<"bank_f32add", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "FP32 grouped addition with an SSA output bank handle";
  let arguments = (ins I64:$a, I64:$b, I64:$c,
                       I64Attr:$rows, I64Attr:$group, BoolAttr:$first);
  let results = (outs I64:$cOut);
  let hasVerifier = 1;
  let assemblyFormat = "$a $b $c prop-dict attr-dict `:` "
                       "type($a) type($b) type($c)";
}

#endif
