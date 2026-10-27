//===- im2colOps.h - Im2colBall operations --------------------------------===//

#ifndef IM2COL_OPS_H
#define IM2COL_OPS_H

#include "Interfaces/BankInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "im2colDialect.h.inc"

#define GET_OP_CLASSES
#include "im2colOps.h.inc"

#endif // IM2COL_OPS_H
