// FP32 erf lowering follows newlib sf_erf.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// expf range reduction and its table follow newlib sf_exp.c/sf_exp2_data.c.
// Copyright (c) 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. The name of the company may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY ARM LTD "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL ARM LTD BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

Value lowerErf(OpBuilder &b, Location loc, Value x) {
  if (auto type = dyn_cast<VectorType>(x.getType())) {
    auto integers = type.clone(b.getI32Type());
    auto fp = [&](uint32_t word) -> Value {
      auto value = b.getFloatAttr(
          b.getF32Type(), APFloat(APFloat::IEEEsingle(), APInt(32, word)));
      return arith::ConstantOp::create(b, loc, type,
                                       DenseElementsAttr::get(type, value));
    };
    auto integer = [&](uint32_t word) -> Value {
      return arith::ConstantOp::create(
          b, loc, integers,
          DenseElementsAttr::get(integers, b.getI32IntegerAttr(word)));
    };
    Value bits = arith::BitcastOp::create(b, loc, integers, x);
    Value absoluteBits =
        arith::AndIOp::create(b, loc, bits, integer(0x7fffffff));
    auto below = [&](uint32_t word) -> Value {
      return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                   absoluteBits, integer(word));
    };
    auto choose = [&](Value condition, Value yes, Value no) -> Value {
      return arith::SelectOp::create(b, loc, condition, yes, no);
    };
    auto fma = [&](Value a, Value c, Value d) -> Value {
      return math::FmaOp::create(b, loc, a, c, d);
    };
    auto polynomial = [&](Value argument, ArrayRef<uint32_t> coefficients) {
      Value result = fp(coefficients.back());
      for (uint32_t word : llvm::reverse(coefficients.drop_back()))
        result = fma(argument, result, fp(word));
      return result;
    };
    Value smallMask = below(0x3f580000);
    Value smallX = choose(smallMask, x, fp(0));
    Value square = arith::MulFOp::create(b, loc, smallX, smallX);
    Value p = polynomial(
        square, {0x3e0375d4, 0xbea66beb, 0xbce9528f, 0xbbbd1489, 0xb7c756b1});
    Value q = polynomial(square, {0x3f800000, 0x3ecbbbce, 0x3d852a63,
                                  0x3ba68116, 0x390aee49, 0xb684e21a});
    Value small = fma(smallX, arith::DivFOp::create(b, loc, p, q), smallX);
    Value tiny = fma(fp(0x3e0375d4), smallX, smallX);
    Value subnormal = arith::MulFOp::create(
        b, loc, fp(0x3e000000),
        fma(fp(0x3f8375d4), smallX,
            arith::MulFOp::create(b, loc, fp(0x41000000), smallX)));
    small = choose(below(0x31800000),
                   choose(below(0x04000000), subnormal, tiny), small);
    Value allSmall = vector::ReductionOp::create(
        b, loc, vector::CombiningKind::AND, smallMask);
    auto middle = scf::IfOp::create(b, loc, TypeRange{type}, allSmall, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&middle.getThenRegion().front());
      scf::YieldOp::create(b, loc, small);
      b.setInsertionPointToStart(&middle.getElseRegion().front());
      Value absolute = arith::BitcastOp::create(b, loc, type, absoluteBits);
      Value offset = arith::SubFOp::create(
          b, loc, choose(below(0x3fa00000), absolute, fp(0x3f800000)),
          fp(0x3f800000));
      Value p =
          polynomial(offset, {0xbb1acdc6, 0x3ed46805, 0xbebe9208, 0x3ea2fe54,
                              0xbde31cc2, 0x3d1151b3, 0xbb0df9c0});
      Value q =
          polynomial(offset, {0x3f800000, 0x3dd9f331, 0x3f0a5785, 0x3d931ae7,
                              0x3e013307, 0x3c5f6e13, 0x3c445aa3});
      Value value = arith::AddFOp::create(b, loc, fp(0x3f58560b),
                                          arith::DivFOp::create(b, loc, p, q));
      Value negative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                             bits, integer(0));
      value = choose(negative, arith::NegFOp::create(b, loc, value), value);
      scf::YieldOp::create(b, loc, choose(smallMask, small, value));
    }
    b.setInsertionPointAfter(middle);
    Value ordinaryMask = below(0x3fa00000);
    Value allOrdinary = vector::ReductionOp::create(
        b, loc, vector::CombiningKind::AND, ordinaryMask);
    Value inputs, outputs;
    {
      OpBuilder::InsertionGuard guard(b);
      auto function =
          b.getInsertionBlock()->getParent()->getParentOfType<func::FuncOp>();
      b.setInsertionPointToStart(&function.getBody().front());
      auto buffer = MemRefType::get(type.getShape(), b.getF32Type());
      inputs = memref::AllocaOp::create(b, loc, buffer);
      outputs = memref::AllocaOp::create(b, loc, buffer);
    }
    auto far = scf::IfOp::create(b, loc, TypeRange{type}, allOrdinary, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&far.getThenRegion().front());
      scf::YieldOp::create(b, loc, middle.getResult(0));
      b.setInsertionPointToStart(&far.getElseRegion().front());
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      vector::StoreOp::create(b, loc, x, inputs, ValueRange{zero});
      vector::StoreOp::create(b, loc, middle.getResult(0), outputs,
                              ValueRange{zero});
      auto loop = scf::ForOp::create(
          b, loc, zero,
          arith::ConstantIndexOp::create(b, loc, type.getNumElements()),
          arith::ConstantIndexOp::create(b, loc, 1));
      b.setInsertionPointToStart(loop.getBody());
      Value at = loop.getInductionVar();
      Value input = memref::LoadOp::create(b, loc, inputs, ValueRange{at});
      Value bits = arith::BitcastOp::create(b, loc, b.getI32Type(), input);
      Value magnitude = arith::AndIOp::create(
          b, loc, bits, arith::ConstantIntOp::create(b, loc, 0x7fffffff, 32));
      Value selected = arith::CmpIOp::create(
          b, loc, arith::CmpIPredicate::uge, magnitude,
          arith::ConstantIntOp::create(b, loc, 0x3fa00000, 32));
      auto scalar = scf::IfOp::create(b, loc, selected, false);
      b.setInsertionPointToStart(&scalar.getThenRegion().front());
      Value value = lowerErf(b, loc, input);
      memref::StoreOp::create(b, loc, value, outputs, ValueRange{at});
      b.setInsertionPointAfter(loop);
      Value values =
          vector::LoadOp::create(b, loc, type, outputs, ValueRange{zero});
      scf::YieldOp::create(b, loc, values);
    }
    b.setInsertionPointAfter(far);
    return far.getResult(0);
  }
  Type f32 = x.getType(), f64 = b.getF64Type();
  Type i32 = b.getI32Type(), i64 = b.getI64Type();
  auto constant = [&](Type type, TypedAttr value) -> Value {
    return arith::ConstantOp::create(b, loc, type, value);
  };
  auto fp = [&](uint32_t bits) -> Value {
    return constant(
        f32, b.getFloatAttr(b.getF32Type(),
                            APFloat(APFloat::IEEEsingle(), APInt(32, bits))));
  };
  auto wide = [&](double value) -> Value {
    return constant(f64, b.getF64FloatAttr(value));
  };
  auto integer = [&](Type type, uint64_t value) -> Value {
    return constant(type, b.getIntegerAttr(type, value));
  };
  auto add = [&](Value a, Value c) -> Value {
    return arith::AddFOp::create(b, loc, a, c);
  };
  auto sub = [&](Value a, Value c) -> Value {
    return arith::SubFOp::create(b, loc, a, c);
  };
  auto mul = [&](Value a, Value c) -> Value {
    return arith::MulFOp::create(b, loc, a, c);
  };
  auto div = [&](Value a, Value c) -> Value {
    return arith::DivFOp::create(b, loc, a, c);
  };
  auto fma = [&](Value a, Value c, Value d) -> Value {
    return math::FmaOp::create(b, loc, a, c, d);
  };
  auto choose = [&](Value condition, Value yes, Value no) -> Value {
    return arith::SelectOp::create(b, loc, condition, yes, no);
  };
  auto polynomial = [&](Value argument, ArrayRef<uint32_t> coefficients) {
    Value result = fp(coefficients.back());
    for (uint32_t word : llvm::reverse(coefficients.drop_back()))
      result = fma(argument, result, fp(word));
    return result;
  };
  Value bits = arith::BitcastOp::create(b, loc, i32, x);
  Value absoluteBits =
      arith::AndIOp::create(b, loc, bits, integer(i32, 0x7fffffff));
  auto below = [&](uint32_t word) -> Value {
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                 absoluteBits, integer(i32, word));
  };
  Value negative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                         bits, integer(i32, 0));
  Value one = fp(0x3f800000);
  Value absolute = arith::BitcastOp::create(b, loc, f32, absoluteBits);
  // Branch before computing a polynomial so inactive intervals do no work.
  auto branch = [&](Value condition, auto yes, auto no) -> Value {
    auto choice = scf::IfOp::create(b, loc, TypeRange{f32}, condition, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&choice.getThenRegion().front());
      scf::YieldOp::create(b, loc, yes());
      b.setInsertionPointToStart(&choice.getElseRegion().front());
      scf::YieldOp::create(b, loc, no());
    }
    return choice.getResult(0);
  };
  auto smallValue = [&]() -> Value {
    Value square = mul(x, x);
    Value p = polynomial(
        square, {0x3e0375d4, 0xbea66beb, 0xbce9528f, 0xbbbd1489, 0xb7c756b1});
    Value q = polynomial(square, {0x3f800000, 0x3ecbbbce, 0x3d852a63,
                                  0x3ba68116, 0x390aee49, 0xb684e21a});
    Value small = fma(x, div(p, q), x);
    Value tiny = fma(fp(0x3e0375d4), x, x);
    Value subnormal =
        mul(fp(0x3e000000), fma(fp(0x3f8375d4), x, mul(fp(0x41000000), x)));
    small = choose(below(0x31800000),
                   choose(below(0x04000000), subnormal, tiny), small);

    return small;
  };
  auto middleValue = [&]() -> Value {
    Value offset = sub(absolute, one);
    Value p =
        polynomial(offset, {0xbb1acdc6, 0x3ed46805, 0xbebe9208, 0x3ea2fe54,
                            0xbde31cc2, 0x3d1151b3, 0xbb0df9c0});
    Value q =
        polynomial(offset, {0x3f800000, 0x3dd9f331, 0x3f0a5785, 0x3d931ae7,
                            0x3e013307, 0x3c5f6e13, 0x3c445aa3});
    Value middle = add(fp(0x3f58560b), div(p, q));
    middle = choose(negative, arith::NegFOp::create(b, loc, middle), middle);

    return middle;
  };
  auto farValue = [&]() -> Value {
    Value farX = absolute;
    Value reciprocalSquare = div(one, mul(farX, farX));
    Value ratio = branch(
        below(0x4036db6e),
        [&]() -> Value {
          Value pa =
              polynomial(reciprocalSquare,
                         {0xbc21a093, 0xbf31a0b7, 0xc128f022, 0xc2798057,
                          0xc322658c, 0xc3389ae7, 0xc2a2932b, 0xc11d077e});
          Value qa = polynomial(reciprocalSquare,
                                {0x3f800000, 0x419d35ce, 0x4309a863, 0x43d9486f,
                                 0x442158c9, 0x43d6810b, 0x42d9451f, 0x40d23f7c,
                                 0xbd777f97});
          return div(pa, qa);
        },
        [&]() -> Value {
          Value pb = polynomial(reciprocalSquare,
                                {0xbc21a092, 0xbf4c9dd4, 0xc18e104b, 0xc320a2ea,
                                 0xc41f6441, 0xc480230b, 0xc3f1c275});
          Value qb =
              polynomial(reciprocalSquare,
                         {0x3f800000, 0x41f2b459, 0x43a2e571, 0x44c01759,
                          0x4547fdbb, 0x451f90ce, 0x43ed43a7, 0xc1b38712});
          return div(pb, qb);
        });
    Value truncated = arith::BitcastOp::create(
        b, loc, f32,
        arith::AndIOp::create(b, loc,
                              arith::BitcastOp::create(b, loc, i32, farX),
                              integer(i32, 0xfffff000)));

    auto tableType = MemRefType::get({32}, b.getI64Type());
    auto module =
        b.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
    if (!module.lookupSymbol<memref::GlobalOp>("exp2f_table")) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      SmallVector<APInt> table;
      for (uint64_t word :
           {0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
            0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
            0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
            0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
            0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
            0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
            0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
            0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
            0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
            0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
            0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL})
        table.emplace_back(64, word);
      memref::GlobalOp::create(
          b, loc, "exp2f_table", b.getStringAttr("private"), tableType,
          DenseIntElementsAttr::get(RankedTensorType::get({32}, b.getI64Type()),
                                    table),
          true, IntegerAttr());
    }
    Value table = memref::GetGlobalOp::create(b, loc, tableType, "exp2f_table");
    auto exp = [&](Value argument) -> Value {
      Value extended = arith::ExtFOp::create(b, loc, f64, argument);
      Value invLn2 = wide(0x1.71547652b82fep+5);
      Value rounded = fma(invLn2, extended, wide(0x1.8p+52));
      Value k = arith::BitcastOp::create(b, loc, i64, rounded);
      Value remainder =
          fma(invLn2, extended,
              arith::NegFOp::create(b, loc, sub(rounded, wide(0x1.8p+52))));
      Value indices = arith::AndIOp::create(b, loc, k, integer(i64, 31));
      Value at = arith::IndexCastOp::create(b, loc, b.getIndexType(), indices);
      Value lookup = memref::LoadOp::create(b, loc, table, ValueRange{at});
      Value scale = arith::BitcastOp::create(
          b, loc, f64,
          arith::AddIOp::create(
              b, loc, lookup,
              arith::ShLIOp::create(b, loc, k, integer(i64, 47))));
      Value high = fma(wide(0x1.c6af84b912394p-20), remainder,
                       wide(0x1.ebfce50fac4f3p-13));
      Value low = fma(wide(0x1.62e42ff0c52d6p-6), remainder, wide(1));
      Value value = mul(fma(high, mul(remainder, remainder), low), scale);
      return arith::TruncFOp::create(b, loc, f32, value);
    };
    Value first = exp(fma(arith::NegFOp::create(b, loc, truncated), truncated,
                          fp(0xbf100000)));
    Value second = exp(fma(sub(truncated, farX), add(truncated, farX), ratio));
    Value complement = div(mul(first, second), farX);
    Value far = choose(negative, sub(complement, one), sub(one, complement));

    return far;
  };
  auto saturated = [&]() -> Value {
    return choose(negative, fp(0xbf800000), one);
  };
  return branch(
      below(0x7f800000),
      [&]() -> Value {
        return branch(below(0x3f580000), smallValue, [&]() -> Value {
          return branch(below(0x3fa00000), middleValue, [&]() -> Value {
            return branch(below(0x40c00000), farValue, saturated);
          });
        });
      },
      [&]() -> Value { return add(saturated(), div(one, x)); });
}
