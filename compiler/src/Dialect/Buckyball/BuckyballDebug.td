//===-- BuckyballDebug.td - Buckyball debug operations ---- tablegen --===//

#ifndef BUCKYBALL_DEBUG_OPS
#define BUCKYBALL_DEBUG_OPS

def PrintMemRefOp : Buckyball_Op<"print_memref"> {
  let summary = "Print memref contents for debugging";
  let description = [{
    Print the contents of a memref for debugging purposes.
  }];
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintScalarOp : Buckyball_Op<"print_scalar"> {
  let summary = "Print scalar value for debugging";
  let description = [{
    Print a scalar value for debugging purposes.
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // BUCKYBALL_DEBUG_OPS
