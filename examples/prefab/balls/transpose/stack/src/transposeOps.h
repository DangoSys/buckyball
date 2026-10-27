//===- transposeOps.h - TransposeBall operations --------------------------===//

#ifndef TRANSPOSE_OPS_H
#define TRANSPOSE_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "transposeDialect.h.inc"

#define GET_OP_CLASSES
#include "transposeOps.h.inc"

#endif // TRANSPOSE_OPS_H
