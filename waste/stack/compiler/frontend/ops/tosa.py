# ===- tosa.py -----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# The registry of mappings from Buddy Graph to MLIR tosa dialect operations.
#
# ===---------------------------------------------------------------------------

import array
from collections.abc import Sequence
import buddy_mlir.ir as ir
import numpy
from buddy_mlir.dialects import (
    affine,
    arith,
    bufferization,
    linalg,
    math,
    memref,
    scf,
    tensor,
    tosa,
    vector,
)
from buddy_mlir.dialects import (
    complex as complex_dialect,
)
from buddy_mlir.ir import F32Type, IndexType
from buddy.compiler.graph import (
    AbsOp,
    AdaptiveAvgPool1dOp,
    AdaptiveAvgPool2dBackwardOp,
    AdaptiveAvgPool2dOp,
    AdaptiveAvgPool3dOp,
    AdaptiveMaxPool1dOp,
    AdaptiveMaxPool2dOp,
    AddbmmOp,
    AddCMulOp,
    AddMMOp,
    AddOp,
    AddScalarOp,
    AliasOp,
    AllDimsOp,
    AllOp,
    AmaxOp,
    AminOp,
    AnyDimsOp,
    AnyOp,
    ArangeStartStepOp,
    ArgMaxOp,
    ArgMinOp,
    AvgPool1dOp,
    AvgPool2dBackwardOp,
    AvgPool2dOp,
    BaddbmmOp,
    BatchMatmulOp,
    BitwiseAndScalarOp,
    BitwiseAndTensorOp,
    BitwiseLeftShiftOp,
    BitwiseNotOp,
    BitwiseOrOp,
    BitwiseOrScalarOp,
    BitwiseRightShiftOp,
    BitwiseXorOp,
    BitwiseXorScalarOp,
    CauchyOp,
    CdistForwardOp,
    CeilOp,
    ClampMaxOp,
    ClampMaxTensorOp,
    ClampMinOp,
    ClampMinTensorOp,
    ClampOp,
    ClampTensorOp,
    CloneOp,
    ConstantPadNdOp,
    Conv2dOp,
    ConvertElementTypeOp,
    ConvolutionBackwardOp,
    CopysignOp,
    CummaxOp,
    CumminOp,
    DiagonalOp,
    DigammaOp,
    DivOp,
    DivScalarModeOp,
    DivScalarOp,
    DivTensorModeOp,
    EmbeddingBagOp,
    EmbeddingOp,
    EmptyStridedOp,
    EqTensorOp,
    ErfcOp,
    ErfinvOp,
    ErfOp,
    ExpandOp,
    ExpOp,
    FillScalarOp,
    FlashAttentionForCpuPrefillOp,
    FlipOp,
    FloorDivideOp,
    FloorOp,
    FmodOp,
    FrexpOp,
    FullLikeOp,
    GeTensorOp,
    GluOp,
    GQAAttentionFusedOp,
    GtOp,
    GtTensorOp,
    HardswishOp,
    HypotOp,
    I0Op,
    IgammacOp,
    IgammaOp,
    Int4UnpackOp,
    IotaOp,
    IsInfOp,
    IsNanOp,
    LeOp,
    LerpOp,
    LeScalarOp,
    LeTensorOp,
    LgammaOp,
    LocalScalarDenseOp,
    LogicalAndOp,
    LogicalNotOp,
    LogicalOrOp,
    LogicalXorOp,
    LogOp,
    LtScalarOp,
    LtTensorOp,
    MaskedFillOp,
    MaskedScatterOp,
    MaxDimOp,
    MaximumOp,
    MaxPool1dOp,
    MaxPool2dOp,
    MeanDefaultOp,
    MeanOp,
    MinDimOp,
    MinimumOp,
    MulOp,
    NativeDropoutOp,
    NativeGroupNormBackwardOp,
    NativeGroupNormOp,
    NativeLayerNormBackwardOp,
    NegOp,
    NeScalarOp,
    NeTensorOp,
    NextafterOp,
    NormScalarOp,
    NormScalarOptDimOp,
    OnesLikeOp,
    PermuteOp,
    PowScalarOp,
    PowTensorTensorOp,
    ProdOp,
    RandIntLowOp,
    RandpermOp,
    ReciprocalOp,
    ReflectionPad1dOp,
    ReflectionPad2dOp,
    ReflectionPad3dOp,
    ReluOp,
    RemainderOp,
    RepeatOp,
    ReplicationPad2dOp,
    ReplicationPad3dOp,
    ReshapeOp,
    ResizeOp,
    RevOp,
    RreluWithNoiseOp,
    RsqrtOp,
    ScaledDotProductFlashAttentionForCpuOp,
    SelectOp,
    SigmoidOp,
    SignbitOp,
    SignOp,
    SiluOp,
    SliceOp,
    SoftplusOp,
    SpecialAiryAiOp,
    SpecialBesselJ0Op,
    SpecialBesselJ1Op,
    SpecialBesselY0Op,
    SpecialBesselY1Op,
    SpecialChebyshevPolynomialTOp,
    SpecialChebyshevPolynomialUOp,
    SpecialChebyshevPolynomialVOp,
    SpecialChebyshevPolynomialWOp,
    SpecialEntrOp,
    SpecialErfcxOp,
    SpecialHermitePolynomialHeOp,
    SpecialHermitePolynomialHOp,
    SpecialI0eOp,
    SpecialI1eOp,
    SpecialI1Op,
    SpecialLaguerrePolynomialLOp,
    SpecialLegendrePolynomialPOp,
    SpecialLogNdtrOp,
    SpecialModifiedBesselK0Op,
    SpecialModifiedBesselK1Op,
    SpecialNdtriOp,
    SpecialNdtrOp,
    SpecialScaledModifiedBesselK0Op,
    SpecialScaledModifiedBesselK1Op,
    SpecialShiftedChebyshevPolynomialTOp,
    SpecialShiftedChebyshevPolynomialUOp,
    SpecialShiftedChebyshevPolynomialVOp,
    SpecialShiftedChebyshevPolynomialWOp,
    SpecialSphericalBesselJ0Op,
    SpecialZetaOp,
    SplitWithSizesOp,
    SqueezeDimOp,
    SqueezeDimsOp,
    SqueezeOp,
    StackOp,
    StdCorrectionOp,
    StdDefaultOp,
    StdDimOp,
    SubOp,
    SubScalarOp,
    SumDefaultOp,
    SumDimOp,
    TanhOp,
    TensorDType,
    TileOp,
    TOp,
    TransposeOp,
    UnbindOp,
    UnfoldOp,
    UniformOp,
    UnsqueezeOp,
    VarCorrectionOp,
    VarDefaultOp,
    VarDimOp,
    VarMeanOp,
    ViewDtypeOp,
    WhereOp,
    ZerosLikeOp,
    ZerosOp,
)
from stack.compiler.frontend.layout.provenance import has_nhwc_layout
from stack.compiler.frontend.ops.expf import exp_f32
from stack.compiler.frontend.ops.int8 import conv2d as int8_conv2d_op
from stack.compiler.frontend.ops.int8 import matmul as int8_matmul_op
from stack.compiler.frontend.ops.int8 import quantize_activation as int8_quantize_op
from buddy.compiler.ops.utils import *
from buddy.compiler.ops.tosa import (
    _build_reassociation_attr,
    _create_mul_shift_operand,
    _create_permutation_attr,
    _create_shape_operand,
    _create_zero_point_tensor,
    _create_zero_tensor,
    _gen_arith_binary_op,
    _normalize_reduce_dims,
    _reshape_or_extract_for_complex,
)


def _to_nhwc(val, channels, elem_ty, nchw_meta=None):
    """Transpose NCHW->NHWC only when SSA is still NCHW. Fail if layout unknown."""
    shp = [int(x) for x in ir.RankedTensorType(val.type).shape]
    if len(shp) != 4:
        raise RuntimeError(f"expected rank-4 activation, got {shp}")
    # Equal channel/spatial extents cannot identify the axis order by shape.
    if shp[1] == channels and shp[3] == channels:
        if has_nhwc_layout(val):
            n, h, w, c = shp
            return val, n, h, w, c
        n, c, h, w = shp
        val = tosa.TransposeOp(
            ir.RankedTensorType.get([n, h, w, c], elem_ty),
            val,
            _create_permutation_attr([0, 2, 3, 1]),
        ).result
        return val, n, h, w, c
    if nchw_meta is not None:
        n, c, h, w = [int(x) for x in nchw_meta]
        if shp == [n, c, h, w]:
            out_ty = ir.RankedTensorType.get([n, h, w, c], elem_ty)
            val = tosa.TransposeOp(
                out_ty, val, _create_permutation_attr([0, 2, 3, 1])
            ).result
            return val, n, h, w, c
        if shp == [n, h, w, c]:
            return val, n, h, w, c
        raise RuntimeError(
            f"cannot infer NCHW/NHWC for ssa={shp} meta_nchw={[n, c, h, w]}"
        )
    if shp[3] == channels:
        n, h, w, c = shp
        return val, n, h, w, c
    if shp[1] == channels:
        n, c, h, w = shp
        out_ty = ir.RankedTensorType.get([n, h, w, c], elem_ty)
        val = tosa.TransposeOp(
            out_ty, val, _create_permutation_attr([0, 2, 3, 1])
        ).result
        return val, n, h, w, c
    raise RuntimeError(
        f"cannot infer NCHW/NHWC for shape {shp} with channels={channels}"
    )


def _nhwc_to_nchw(value: ir.Value) -> ir.Value:
    shape = [int(x) for x in ir.RankedTensorType(value.type).shape]
    if len(shape) != 4:
        raise RuntimeError(f"_nhwc_to_nchw expects rank-4, got {shape}")
    n, h, w, c = shape
    elem = ir.RankedTensorType(value.type).element_type
    out_ty = ir.RankedTensorType.get([n, c, h, w], elem)
    return tosa.TransposeOp(
        out_ty, value, _create_permutation_attr([0, 3, 1, 2])
    ).result


def _nchw_to_nhwc(value: ir.Value) -> ir.Value:
    shape = [int(x) for x in ir.RankedTensorType(value.type).shape]
    if len(shape) != 4:
        raise RuntimeError(f"_nchw_to_nhwc expects rank-4, got {shape}")
    n, c, h, w = shape
    elem = ir.RankedTensorType(value.type).element_type
    out_ty = ir.RankedTensorType.get([n, h, w, c], elem)
    return tosa.TransposeOp(
        out_ty, value, _create_permutation_attr([0, 2, 3, 1])
    ).result


def _split_via_tosa_slice(
    input_tensor: ir.Value,
    input_shape: list[int],
    dim: int,
    split_sizes: list[int],
) -> list[ir.Value]:
    rank = len(input_shape)
    elem = ir.RankedTensorType(input_tensor.type).element_type
    results = []
    offset = 0
    for size in split_sizes:
        start = [0] * rank
        start[dim] = offset
        out_shape = list(input_shape)
        out_shape[dim] = size
        out_ty = ir.RankedTensorType.get(out_shape, elem)
        results.append(
            tosa.SliceOp(
                out_ty,
                input_tensor,
                _create_shape_operand(start),
                _create_shape_operand(out_shape),
            ).result
        )
        offset += size
    return results


def _align_binary_layout(input1, input2):
    """Keep NHWC activations and align logical NCHW broadcast operands."""
    if not (has_nhwc_layout(input1) or has_nhwc_layout(input2)):
        return input1, input2
    operands = [input1, input2]
    for index, value in enumerate(operands):
        if not isinstance(value, ir.Value) or has_nhwc_layout(value):
            continue
        shape = list(ir.RankedTensorType(value.type).shape)
        shape = [1] * (4 - len(shape)) + shape
        value = tosa.ReshapeOp(value, _create_shape_operand(shape)).result
        if sum(size != 1 for size in shape) <= 1:
            value = tosa.ReshapeOp(
                value,
                _create_shape_operand([shape[0], shape[2], shape[3], shape[1]]),
            ).result
        else:
            value = _nchw_to_nhwc(value)
        value.owner.attributes["buddy.layout"] = ir.StringAttr.get("nhwc")
        operands[index] = value
    return operands


def add_op(node: AddOp, symbol_table):
    """
    Import tensor addition operation.
    From buddy graph ir's `AddOp` operator to MLIR TOSA `add` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if isinstance(node.args[0], str) and isinstance(node.args[1], str):
        input1_dtype = ir.RankedTensorType(input1.type).element_type
        input2_dtype = ir.RankedTensorType(input2.type).element_type
        if input1_dtype != mlir_dtype:
            input1 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input1.type).shape,
                    mlir_dtype,
                ),
                input1,
            ).result
        if input2_dtype != mlir_dtype:
            input2 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input2.type).shape,
                    mlir_dtype,
                ),
                input2,
            ).result
    input1, input2 = _align_binary_layout(input1, input2)
    return _gen_arith_binary_op(input1, input2, tosa.AddOp)


def sub_op(node: SubOp, symbol_table):
    """
    Import tensor subtraction operation.
    From buddy graph ir's `SubOp` operator to MLIR TOSA `sub` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if isinstance(node.args[0], str) and isinstance(node.args[1], str):
        input1_dtype = ir.RankedTensorType(input1.type).element_type
        input2_dtype = ir.RankedTensorType(input2.type).element_type
        if input1_dtype != mlir_dtype:
            input1 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input1.type).shape,
                    mlir_dtype,
                ),
                input1,
            ).result
        if input2_dtype != mlir_dtype:
            input2 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input2.type).shape,
                    mlir_dtype,
                ),
                input2,
            ).result
    input1, input2 = _align_binary_layout(input1, input2)
    return _gen_arith_binary_op(input1, input2, tosa.SubOp)


def mul_op(node: MulOp, symbol_table):
    """
    Import tensor division operation.
    From buddy graph ir's `DivOp` operator to MLIR TOSA `div` operation.
    """

    def _inner_op(result_type, input1, input2):
        shift = _create_mul_shift_operand()
        return tosa.MulOp(result_type, input1, input2, shift)

    output_shape = list(node.tensor_meta["shape"])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)

    if isinstance(node.args[0], str):
        input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    else:
        data = [node.args[0]]
        input1_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input1_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[0])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result

    if isinstance(node.args[1], str):
        input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    else:
        data = [node.args[1]]
        input2_shape = numpy.array(data).shape
        tensor_type = ir.RankedTensorType.get(input2_shape, mlir_dtype)
        element = mlir_element_attr_get(dtype, node.args[1])
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        input2 = arith.ConstantOp(tensor_type, attr).result

    input1_dtype = ir.RankedTensorType(input1.type).element_type
    input2_dtype = ir.RankedTensorType(input2.type).element_type
    if input1_dtype != mlir_dtype:
        input1 = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(input1.type).shape,
                mlir_dtype,
            ),
            input1,
        ).result
    if input2_dtype != mlir_dtype:
        input2 = tosa.CastOp(
            ir.RankedTensorType.get(
                ir.RankedTensorType(input2.type).shape,
                mlir_dtype,
            ),
            input2,
        ).result

    input1, input2 = _align_binary_layout(input1, input2)
    return _gen_arith_binary_op(input1, input2, _inner_op)


def div_op(node: DivOp, symbol_table):
    """
    Import tensor division operation.
    From buddy graph ir's `DivOp` operator to MLIR TOSA `div` operation.
    """

    def _inner_op(result_type, input1, input2):
        # Preserve true division. Reciprocal followed by multiplication can
        # cross round-to-even boundaries in activation quantization.
        shape = list(result_type.shape)
        rank = len(shape)
        maps = []
        for value in (input1, input2):
            dims = [
                (
                    ir.AffineExpr.get_constant(0)
                    if extent == 1
                    else ir.AffineExpr.get_dim(i)
                )
                for i, extent in enumerate(ir.RankedTensorType(value.type).shape)
            ]
            maps.append(ir.AffineMapAttr.get(ir.AffineMap.get(rank, 0, dims)))
        maps.append(ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank)))
        empty = tensor.EmptyOp(shape, result_type.element_type)
        division = linalg.GenericOp(
            [result_type],
            [input1, input2],
            [empty.result],
            ir.ArrayAttr.get(maps),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
            ),
        )
        block = ir.Block.create_at_start(
            division.region, [result_type.element_type] * 3
        )
        with ir.InsertionPoint(block):
            value = arith.DivFOp(block.arguments[0], block.arguments[1]).result
            linalg.YieldOp([value])
        if has_nhwc_layout(input1) or has_nhwc_layout(input2):
            division.operation.attributes["buddy.layout"] = ir.StringAttr.get("nhwc")
        return division

    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])

    # PyTorch integer division returns float32. Cast any integer tensor inputs to
    # float32 so tosa.reciprocal is legalizable and the result type matches Torch
    # semantics (true division).
    def _is_int_tensor(value: object) -> bool:
        if not isinstance(value, ir.Value):
            return False
        element_type = ir.RankedTensorType(value.type).element_type
        return isinstance(element_type, ir.IntegerType)

    if _is_int_tensor(input1) or _is_int_tensor(input2):
        f32 = ir.F32Type.get()
        if isinstance(input1, ir.Value):
            in1_ty = ir.RankedTensorType.get(
                list(ir.RankedTensorType(input1.type).shape), f32
            )
            input1 = tosa.CastOp(in1_ty, input1).result
        elif isinstance(input1, int):
            input1 = float(input1)
        if isinstance(input2, ir.Value):
            in2_ty = ir.RankedTensorType.get(
                list(ir.RankedTensorType(input2.type).shape), f32
            )
            input2 = tosa.CastOp(in2_ty, input2).result
        elif isinstance(input2, int):
            input2 = float(input2)

    input1, input2 = _align_binary_layout(input1, input2)
    return _gen_arith_binary_op(input1, input2, _inner_op)


def amax_op(node: AmaxOp, symbol_table):
    """
    Import the amax operation.
    From buddy graph ir's `AmaxOp` operator to MLIR TOSA `reduce_max`
    operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    dim_arg = node.args[1] if len(node.args) > 1 else None
    keepdim = node.args[2] if len(node.args) > 2 else False
    rank = len(ir.RankedTensorType(input1.type).shape)
    dims = _normalize_reduce_dims(dim_arg, rank)
    input_nhwc = has_nhwc_layout(input1)
    if input_nhwc:
        dims = [[0, 3, 1, 2][axis] for axis in dims]

    result = input1
    for dim_val in dims:
        dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), dim_val)
        result = tosa.ReduceMaxOp(result, dim_attr).result

    if input_nhwc:
        shape = list(ir.RankedTensorType(result.type).shape)
        if sum(size != 1 for size in shape) <= 1:
            logical_shape = [shape[0], shape[3], shape[1], shape[2]]
            result = tosa.ReshapeOp(result, _create_shape_operand(logical_shape)).result
        else:
            result = _nhwc_to_nchw(result)
    if not keepdim:
        output_shape = list(node.tensor_meta["shape"])
        reshape_operand = _create_shape_operand(output_shape)
        result = tosa.ReshapeOp(result, reshape_operand).result

    return result


def avg_pool2d_op(node: AvgPool2dOp, symbol_table):
    """
    Import the avg_pool2d operation.
    From Buddy AvgPool2dOp to MLIR TOSA `avg_pool2d` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    kernel = node.args[1]
    stride = node.args[2]
    if len(node.args) > 3:
        pad = node.args[3]
    else:
        pad = [0 for _ in kernel]
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    acc_type = ir.TypeAttr.get(result_element_type)

    meta = list(node.tensor_meta["shape"])
    channels = int(meta[1])
    src = node.args[0]
    nchw_meta = (
        list(src.tensor_meta["shape"])
        if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
        else None
    )
    input1, N, H, W, C = _to_nhwc(input1, channels, result_element_type, nchw_meta)
    input_shape_nhwc = [N, H, W, C]

    # Expand pad to 4 elements if needed
    if len(pad) == 1:
        pad = [pad[0]] * 4
    elif len(pad) == 2:
        pad = [pad[0]] * 2 + [pad[1]] * 2

    # Calculate output shape in NHWC format
    # Formula: output = (input + pad_top + pad_bottom - kernel) / stride + 1
    pad_top, pad_bottom, pad_left, pad_right = pad
    output_h = (H + pad_top + pad_bottom - kernel[0]) // stride[0] + 1
    output_w = (W + pad_left + pad_right - kernel[1]) // stride[1] + 1
    output_shape_nhwc = [N, output_h, output_w, C]

    kernel_attr = ir._denseI64ArrayAttr(kernel, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    pad_attr = ir._denseI64ArrayAttr(pad, None)

    output = ir.RankedTensorType.get(output_shape_nhwc, result_element_type)
    input_zp = _create_zero_point_tensor(input1)
    output_zp = _create_zero_point_tensor(input1)
    op = tosa.AvgPool2dOp(
        output,
        input1,
        input_zp,
        output_zp,
        kernel=kernel_attr,
        stride=stride_attr,
        pad=pad_attr,
        acc_type=acc_type,
    )
    return op


def adaptive_avg_pool2d_op(node: AdaptiveAvgPool2dOp, symbol_table):
    """
    Import the adaptive_avg_pool2d operation.
    From buddy graph ir's `AdaptiveAvgPool2dOp` operator to MLIR operations.
    Uses TOSA avg_pool2d with calculated kernel/stride.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    input_is_nhwc = has_nhwc_layout(input1)
    output_size = node.args[1]
    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    acc_type = ir.TypeAttr.get(result_element_type)

    # Get input shape; sticky NHWC after conv keeps channels last.
    channels = int(list(node.tensor_meta["shape"])[1])
    src = node.args[0]
    nchw_meta = (
        list(src.tensor_meta["shape"])
        if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
        else None
    )
    input1, N, H, W, C = _to_nhwc(input1, channels, result_element_type, nchw_meta)
    out_h, out_w = output_size

    # Calculate kernel and stride for adaptive pooling
    kernel_h = (H + out_h - 1) // out_h
    kernel_w = (W + out_w - 1) // out_w
    stride_h = H // out_h
    stride_w = W // out_w

    # Apply avg_pool2d
    kernel_attr = ir._denseI64ArrayAttr([kernel_h, kernel_w], None)
    stride_attr = ir._denseI64ArrayAttr([stride_h, stride_w], None)
    pad_attr = ir._denseI64ArrayAttr([0, 0, 0, 0], None)

    pool_nhwc_shape = [N, out_h, out_w, C]
    pool_type = ir.RankedTensorType.get(pool_nhwc_shape, result_element_type)
    input_zp = _create_zero_point_tensor(input1)
    output_zp = _create_zero_point_tensor(input1)
    pooled = tosa.AvgPool2dOp(
        pool_type,
        input1,
        input_zp,
        output_zp,
        kernel=kernel_attr,
        stride=stride_attr,
        pad=pad_attr,
        acc_type=acc_type,
    )
    if input_is_nhwc:
        return pooled
    return tosa.TransposeOp(
        ir.RankedTensorType.get([N, C, out_h, out_w], result_element_type),
        pooled.result,
        _create_permutation_attr([0, 3, 1, 2]),
    )


def _reshape_input_nchw_meta(node, symbol_table):
    """PyTorch NCHW shape of reshape/view input. args[0] is often a name str."""
    src = node.args[0]
    if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta:
        return [int(x) for x in src.tensor_meta["shape"]]
    if isinstance(src, str):
        ops = symbol_table.get(("__buddy_ops_by_name__", 0))
        if ops is None:
            return None
        pred = ops.get(src)
        if pred is None or not (
            hasattr(pred, "tensor_meta") and "shape" in pred.tensor_meta
        ):
            return None
        return [int(x) for x in pred.tensor_meta["shape"]]
    return None


def reshape_op(node: ReshapeOp, symbol_table):
    """
    Import the reshape operation.
    From buddy graph ir's `ReshapeOp` operator to MLIR TOSA `reshape`
    operation.

    Note: If the new shape contains one and only one `-1`, the size of the new
    shape will be inferred automatically.
    """
    input1 = symbol_table.get((str(node.args[0]), 0))
    new_shape = []
    if node._newshape is None:
        shape_arg = node.args[1]

        if isinstance(shape_arg, (list, tuple)):
            new_shape = list(shape_arg)
        else:
            try:
                new_shape = list(shape_arg)
            except TypeError:
                new_shape = [shape_arg]
    else:
        new_shape = list(node._newshape)

    now_shape = ir.RankedTensorType(input1.type).shape
    total_size = 1
    for dim_siz in now_shape:
        total_size *= dim_siz

    neg_one_cnt = 0
    rest_size = 1
    for dim_siz in new_shape:
        if dim_siz == -1:
            neg_one_cnt += 1
            continue
        rest_size *= dim_siz

    if neg_one_cnt != 0:
        if neg_one_cnt > 1 or total_size % rest_size != 0:
            raise ValueError("Can not infer the new shape!")
        infer_dim_size = total_size // rest_size
        for i, _ in enumerate(new_shape):
            if new_shape[i] == -1:
                new_shape[i] = infer_dim_size

    producer = input1.owner
    if isinstance(producer, ir.OpView):
        producer = producer.operation
    if len(now_shape) == 4 and isinstance(producer, ir.Operation):
        attrs = producer.attributes
        if (
            (
                "buckyball.mega_conv2d" in attrs
                or "buckyball.mega_conv2d_depthwise" in attrs
            )
            and "final_output" in attrs
            and ir.BoolAttr(attrs["final_output"]).value
            and ir.RankedTensorType(input1.type).element_type == ir.F32Type.get()
        ):
            return _reshape_or_extract_for_complex(input1, new_shape)

    if len(now_shape) == 4 and has_nhwc_layout(input1):
        input_meta = _reshape_input_nchw_meta(node, symbol_table)
        if input_meta is None or len(input_meta) != 4:
            raise RuntimeError("reshape of NHWC activation requires Torch input shape")
        n, c, h, w = input_meta
        if list(now_shape) != [n, h, w, c]:
            raise RuntimeError(
                f"reshape NHWC layout mismatch: ssa={list(now_shape)} torch={input_meta}"
            )
        element_type = ir.RankedTensorType(input1.type).element_type
        input1 = tosa.TransposeOp(
            ir.RankedTensorType.get(input_meta, element_type),
            input1,
            _create_permutation_attr([0, 3, 1, 2]),
        ).result

    return _reshape_or_extract_for_complex(input1, new_shape)


def maxpool2d_op(node: MaxPool2dOp, symbol_table):
    """
    Import the maxpool2d operation.
    From Buddy MaxPool2dOp to MLIR TOSA `max_pool2d` operation.
    """
    if len(node.args) == 5:
        raise NotImplementedError
    input1 = symbol_table.get((str(node.args[0]), 0))
    kernel = node.args[1]
    stride = node.args[2]
    if len(node.args) > 3:
        pad = node.args[3]
    else:
        pad = [0 for _ in kernel]
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)

    original_shape = list(ir.RankedTensorType(input1.type).shape)
    channels = int(list(node.tensor_meta["shape"])[1])
    src = node.args[0]
    nchw_meta = (
        list(src.tensor_meta["shape"])
        if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
        else None
    )
    input1, n, h, w, c = _to_nhwc(input1, channels, result_element_type, nchw_meta)
    in_n, in_c, in_h, in_w = n, c, h, w

    out_shape = node.tensor_meta["shape"]
    if len(pad) == 1:
        pad = [pad[0]] * 4
    elif len(pad) == 2:
        pad = [pad[0]] * 2 + [pad[1]] * 2

    k_h, k_w = kernel
    s_h, s_w = stride
    pt, pb, pl, pr = pad
    # Preserve the leading padding. TOSA requires an exact output extent;
    # adjust only the trailing padding to match Torch floor/ceil semantics.
    out_h, out_w = node.tensor_meta["shape"][2:]
    pb = max(0, (out_h - 1) * s_h + k_h - in_h - pt)
    pr = max(0, (out_w - 1) * s_w + k_w - in_w - pl)

    kernel_attr = ir._denseI64ArrayAttr(kernel, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)
    pad_attr = ir._denseI64ArrayAttr([pt, pb, pl, pr], None)
    out_h = (in_h + pt + pb - k_h) // s_h + 1
    out_w = (in_w + pl + pr - k_w) // s_w + 1
    out_shape_nhwc = [in_n, out_h, out_w, in_c]
    output = ir.RankedTensorType.get(out_shape_nhwc, result_element_type)
    op = tosa.MaxPool2dOp(output, input1, kernel_attr, stride_attr, pad_attr)
    return op


def convolution2d_op(node: Conv2dOp, symbol_table):
    """
    Import the convolution operation.
    From Buddy Conv2dOp to MLIR TOSA `conv2d` operation.
    arg[0]: Tensor input
    arg[1]: Tensor weight
    arg[2]: Tensor? bias
    arg[3]: SymInt[] stride
    arg[4]: SymInt[] padding
    arg[5]: SymInt[] dilation
    arg[6]: bool transposed
    arg[7]: SymInt[] output_padding
    arg[8]: SymInt groups
    """
    # Get arguments from convolution node.
    assert len(node.args) == 9
    input = node.args[0]
    weight = node.args[1]
    bias = node.args[2]
    stride = node.args[3]
    input_padding = node.args[4]
    dilation = node.args[5]
    is_kernel_transposed = node.args[6]
    out_padding = node.args[7]
    groups = node.args[8]

    # Prepare input, weight, and output information.
    input_val = symbol_table.get((str(input), 0))
    input_shape = list(ir.RankedTensorType(input_val.type).shape)
    weight_val = symbol_table.get((str(weight), 0))
    weight_shape = ir.RankedTensorType(weight_val.type).shape
    dtype = node.tensor_meta["dtype"]
    result_element_type = mlir_element_type_get(dtype)
    out_shape = node.tensor_meta["shape"]
    acc_type = ir.TypeAttr.get(result_element_type)

    # Prepare Depthwise Conv2D information
    is_grouped = (list(weight_shape)[1] == 1) and (groups != 1)
    is_depthwise = (groups == list(weight_shape)[0]) and is_grouped

    # Prepare input channel and output channel.
    if is_kernel_transposed:
        in_channels = list(weight_shape)[0]
        out_channels = list(weight_shape)[1] * groups
    else:
        in_channels = list(weight_shape)[1] * groups
        out_channels = list(weight_shape)[0]

    # Prepare bias tensor.
    if len(node._parents) == 2:
        new_size_tensor_type = ir.RankedTensorType.get(
            [out_channels], result_element_type
        )
        element = mlir_element_attr_get(dtype, 0)
        new_size_attr = ir.DenseElementsAttr.get_splat(new_size_tensor_type, element)
        bias_tensor = tosa.ConstOp(new_size_attr).results[0]
    else:
        bias_tensor = symbol_table.get((str(bias), 0))

    # Prepare attributes.
    dilation_attr = ir._denseI64ArrayAttr(dilation, None)
    stride_attr = ir._denseI64ArrayAttr(stride, None)

    # Convolution 2D
    if len(weight_shape) == 4:
        # Prepare input padding.
        if len(input_padding) == 1:
            input_padding = [input_padding[0]] * 4
        elif len(input_padding) == 2:
            input_padding = [input_padding[0]] * 2 + [input_padding[1]] * 2

        t, b, l, r = input_padding
        sy, sx = int(stride[0]), int(stride[1])
        dy, dx = int(dilation[0]), int(dilation[1])
        kernel_h = int(list(weight_shape)[2])
        kernel_w = int(list(weight_shape)[3])

        extra_out_h = 0
        extra_out_w = 0
        src = node.args[0]
        nchw_meta = (
            list(src.tensor_meta["shape"])
            if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
            else None
        )
        input_val, _, input_h, input_w, _ = _to_nhwc(
            input_val, in_channels, result_element_type, nchw_meta
        )
        input_h = int(input_h)
        input_w = int(input_w)

        def _adjust_padding(input_size, kernel, dil, stride_val, pad0, pad1):
            """Ensure stride divisibility by adding pad to the second side."""
            base = input_size - 1 - (kernel - 1) * dil
            total = base + pad0 + pad1
            remainder = total % stride_val
            if remainder == 0:
                return pad0, pad1, 0
            pad_needed = stride_val - remainder
            pad1 += pad_needed
            return pad0, pad1, 1

        if not is_kernel_transposed:
            t, b, extra_h = _adjust_padding(input_h, kernel_h, dy, sy, int(t), int(b))
            l, r, extra_w = _adjust_padding(input_w, kernel_w, dx, sx, int(l), int(r))
            extra_out_h += extra_h
            extra_out_w += extra_w
        input_padding = [t, b, l, r]

        # Prepare input_padding attributes.
        input_padding_attr = ir._denseI64ArrayAttr(input_padding, None)
        # TOSA conv2d is NHWC; keep output in NHWC (no convert-back).
        out_shape = [
            out_shape[0],
            out_shape[2],
            out_shape[3],
            out_shape[1],
        ]
        conv_out_shape = list(out_shape)
        if extra_out_h:
            conv_out_shape[1] += extra_out_h
        if extra_out_w:
            conv_out_shape[2] += extra_out_w
        output_type = ir.RankedTensorType.get(conv_out_shape, result_element_type)

        if (
            str(ir.RankedTensorType(weight_val.type).element_type) == "i8"
            and not is_depthwise
        ):
            if is_kernel_transposed:
                raise ValueError("quantized transpose convolution is unsupported")
            if stride[0] != stride[1]:
                raise ValueError("quantized convolution requires equal strides")
            pad_constant = _create_shape_operand(
                [0, 0, int(t), int(b), int(l), int(r), 0, 0]
            )
            pad_value = _create_zero_point_tensor(input_val)
            padded_type = ir.RankedTensorType.get(
                [
                    conv_out_shape[0],
                    input_h + int(t) + int(b),
                    input_w + int(l) + int(r),
                    in_channels,
                ],
                result_element_type,
            )
            padded_input = tosa.PadOp(
                padded_type, input_val, pad_constant, pad_value
            ).result
            quant_weight = weight_val
            if node._layout.find("FCHW") != -1:
                perm_shape = [
                    weight_shape[0],
                    weight_shape[2],
                    weight_shape[3],
                    weight_shape[1],
                ]
                quant_weight = tosa.TransposeOp(
                    ir.RankedTensorType.get(perm_shape, ir.IntegerType.get_signless(8)),
                    weight_val,
                    _create_permutation_attr([0, 2, 3, 1]),
                ).result
            bias_output = tensor.EmptyOp(conv_out_shape, result_element_type)
            generic_map = ir.AffineMap.get_permutation([0, 1, 2, 3])
            bias_init = linalg.GenericOp(
                [output_type],
                [bias_tensor],
                [bias_output],
                ir.ArrayAttr.get(
                    [
                        ir.AffineMapAttr.get(generic_map.get_submap([3])),
                        ir.AffineMapAttr.get(generic_map),
                    ]
                ),
                ir.ArrayAttr.get(
                    [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4
                ),
            )
            bias_block = ir.Block.create_at_start(
                bias_init.region, [result_element_type, result_element_type]
            )
            bias_init.region.blocks[0].append(linalg.YieldOp([bias_block.arguments[0]]))
            quant_conv = linalg.Conv2DNhwcFhwcOp(
                [output_type],
                [padded_input, quant_weight],
                [bias_init.result],
                strides=stride_attr,
                dilations=dilation_attr,
            )
            linalg.fill_builtin_region(quant_conv.operation)
            i64 = ir.IntegerType.get_signless(64)
            quant_conv.operation.attributes["dw_addr"] = ir.IntegerAttr.get(
                i64, node._dw_addr
            )
            quant_conv.operation.attributes["dw_bytes"] = ir.IntegerAttr.get(
                i64, node._dw_bytes
            )
            quant_conv.operation.attributes["per_channel"] = ir.BoolAttr.get(
                node._per_channel
            )
            quant_conv.operation.attributes["strides"] = ir.DenseElementsAttr.get(
                numpy.array([int(stride[0]), int(stride[1])], dtype=numpy.int64),
                type=ir.VectorType.get([2], i64),
            )
            op = quant_conv
            if extra_out_h or extra_out_w:
                slice_start = _create_shape_operand([0, 0, 0, 0])
                slice_size = _create_shape_operand(out_shape)
                slice_type = ir.RankedTensorType.get(out_shape, result_element_type)
                op = tosa.SliceOp(slice_type, op.result, slice_start, slice_size)
            return op

        # Depthwise Conv2D Operation.
        if is_depthwise is True:
            # If groups == in_channels,out_channels == in_channels
            weight_depthwise = weight_val
            if node._layout.find("FCHW") != -1:
                perm_list = [2, 3, 0, 1]
                perms_attr = _create_permutation_attr(perm_list)
                perm_shape = []
                perm_shape.append(weight_shape[2])
                perm_shape.append(weight_shape[3])
                perm_shape.append(weight_shape[0])
                perm_shape.append(weight_shape[1])
                permute_result_type = ir.RankedTensorType.get(
                    perm_shape,
                    ir.RankedTensorType(weight_val.type).element_type,
                )
                weight_depthwise = tosa.TransposeOp(
                    permute_result_type, weight_val, perms_attr
                ).result
            if str(ir.RankedTensorType(weight_depthwise.type).element_type) == "i8":
                pad_constant = _create_shape_operand(
                    [0, 0, int(t), int(b), int(l), int(r), 0, 0]
                )
                pad_value = _create_zero_point_tensor(input_val)
                padded_shape = [
                    conv_out_shape[0],
                    input_h + int(t) + int(b),
                    input_w + int(l) + int(r),
                    conv_out_shape[3],
                ]
                padded_type = ir.RankedTensorType.get(padded_shape, result_element_type)
                input_val = tosa.PadOp(
                    padded_type, input_val, pad_constant, pad_value
                ).result
                depth_shape = [
                    conv_out_shape[0],
                    conv_out_shape[1],
                    conv_out_shape[2],
                    conv_out_shape[3],
                    1,
                ]
                depth_type = ir.RankedTensorType.get(depth_shape, result_element_type)
                depth_init = tensor.EmptyOp(depth_shape, result_element_type)
                identity = ir.AffineMap.get_permutation([0, 1, 2, 3, 4])
                bias_init = linalg.GenericOp(
                    [depth_type],
                    [bias_tensor],
                    [depth_init],
                    ir.ArrayAttr.get(
                        [
                            ir.AffineMapAttr.get(identity.get_submap([3])),
                            ir.AffineMapAttr.get(identity),
                        ]
                    ),
                    ir.ArrayAttr.get(
                        [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 5
                    ),
                )
                bias_block = ir.Block.create_at_start(
                    bias_init.region, [result_element_type, result_element_type]
                )
                bias_init.region.blocks[0].append(
                    linalg.YieldOp([bias_block.arguments[0]])
                )
                depthwise_op = linalg.DepthwiseConv2DNhwcHwcmOp(
                    [depth_type],
                    [input_val, weight_depthwise],
                    [bias_init.result],
                    strides=stride_attr,
                    dilations=dilation_attr,
                )
                linalg.fill_builtin_region(depthwise_op.operation)
                depthwise_op.operation.attributes["dw_addr"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), node._dw_addr
                )
                depthwise_op.operation.attributes["dw_bytes"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), node._dw_bytes
                )
                depthwise_op.operation.attributes["per_channel"] = ir.BoolAttr.get(
                    node._per_channel
                )
                collapse = tensor.CollapseShapeOp(
                    output_type,
                    depthwise_op.result,
                    _build_reassociation_attr([[0], [1], [2], [3, 4]]),
                ).result
                if extra_out_h or extra_out_w:
                    slice_start = _create_shape_operand([0, 0, 0, 0])
                    slice_size = _create_shape_operand(out_shape)
                    slice_type = ir.RankedTensorType.get(out_shape, result_element_type)
                    return tosa.SliceOp(slice_type, collapse, slice_start, slice_size)
                return collapse
            input_zp = _create_zero_point_tensor(input_val)
            weight_zp = _create_zero_point_tensor(weight_depthwise)
            depthwise_op = tosa.DepthwiseConv2DOp(
                output_type,
                input_val,
                weight_depthwise,
                bias_tensor,
                input_zp,
                weight_zp,
                input_padding_attr,
                stride_attr,
                dilation_attr,
                acc_type,
            )
            if hasattr(node, "_dw_addr"):
                i64 = ir.IntegerType.get_signless(64)
                depthwise_op.operation.attributes["dw_addr"] = ir.IntegerAttr.get(
                    i64, node._dw_addr
                )
                depthwise_op.operation.attributes["dw_bytes"] = ir.IntegerAttr.get(
                    i64, node._dw_bytes
                )
                depthwise_op.operation.attributes["per_channel"] = ir.BoolAttr.get(
                    node._per_channel
                )
            op = depthwise_op
        else:
            # Transpose Conv2D Operation.
            if is_kernel_transposed:
                if sum(input_padding) > 0 or sum(dilation) > len(dilation):
                    raise NotImplementedError
                for i in range(len(out_padding), 4):
                    out_padding = [0] + out_padding
                out_padding_attr = ir._denseI64ArrayAttr(out_padding, None)
                input_zp = _create_zero_point_tensor(input_val)
                weight_zp = _create_zero_point_tensor(weight_val)
                op = tosa.TransposeConv2DOp(
                    output_type,
                    input_val,
                    weight_val,
                    bias_tensor,
                    input_zp,
                    weight_zp,
                    out_padding_attr,
                    stride_attr,
                    acc_type,
                )
            # Generic Conv2D Operation.
            else:
                if node._layout.find("FCHW") != -1:
                    perm_list = [0, 2, 3, 1]
                    perms_attr = _create_permutation_attr(perm_list)
                    perm_shape = []
                    perm_shape.append(weight_shape[0])
                    perm_shape.append(weight_shape[2])
                    perm_shape.append(weight_shape[3])
                    perm_shape.append(weight_shape[1])
                    permute_result_type = ir.RankedTensorType.get(
                        perm_shape,
                        ir.RankedTensorType(weight_val.type).element_type,
                    )
                    weight_val = tosa.TransposeOp(
                        permute_result_type,
                        weight_val,
                        perms_attr,
                    ).result
                input_zp = _create_zero_point_tensor(input_val)
                weight_zp = _create_zero_point_tensor(weight_val)
                conv_op = tosa.Conv2DOp(
                    output_type,
                    input_val,
                    weight_val,
                    bias_tensor,
                    input_zp,
                    weight_zp,
                    input_padding_attr,
                    stride_attr,
                    dilation_attr,
                    acc_type,
                )
                if hasattr(node, "_dw_addr"):
                    i64 = ir.IntegerType.get_signless(64)
                    conv_op.operation.attributes["dw_addr"] = ir.IntegerAttr.get(
                        i64, node._dw_addr
                    )
                    conv_op.operation.attributes["dw_bytes"] = ir.IntegerAttr.get(
                        i64, node._dw_bytes
                    )
                    conv_op.operation.attributes["per_channel"] = ir.BoolAttr.get(
                        node._per_channel
                    )
                op = conv_op

        if extra_out_h or extra_out_w:
            # Slice off the padded rows/cols that exceed the expected output.
            slice_start = _create_shape_operand([0, 0, 0, 0])
            slice_size = _create_shape_operand(out_shape)
            slice_type = ir.RankedTensorType.get(out_shape, result_element_type)
            slice_op = tosa.SliceOp(slice_type, op.result, slice_start, slice_size)
            op = slice_op
    # Convolution 1D
    elif len(weight_shape) == 3:
        # Prepare input with padding.
        if input_padding[0] != 0:
            input_shape = list(ir.RankedTensorType(input_val.type).shape)
            padded_type = ir.RankedTensorType.get(
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * input_padding[0],
                ],
                result_element_type,
            )

            pad_constant = _create_shape_operand(
                [0, 0, 0, 0, input_padding[0], input_padding[0]]
            )
            ty = ir.Type.parse("tensor<1xf32>")
            pad_zp = tosa.ConstOp(
                ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get_f32(0.0))
            ).result
            input_val = tosa.PadOp(padded_type, input_val, pad_constant, pad_zp)
        output_type = ir.RankedTensorType.get(out_shape, result_element_type)
        output_conv = _create_zero_tensor(output_type)
        assert groups == 1, "only support one group"
        # Con1D Operation Without Bias
        conv_op = linalg.conv_1d_ncw_fcw(
            input_val,
            weight_val,
            outs=[output_conv],
            strides=stride_attr,
            dilations=dilation_attr,
        )
        output = tensor.EmptyOp(list(out_shape), result_element_type)
        generic_map = ir.AffineMap.get_permutation(
            [i for i in range(len(list(out_shape)))]
        )
        loop_type = [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(
            list(out_shape)
        )
        loop_type[1] = ir.Attribute.parse("#linalg.iterator_type<reduction>")
        # Add Bias To Conv2d.
        op = linalg.GenericOp(
            [output_type],
            [conv_op, bias_tensor],
            [output],
            ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(
                        generic_map.get_submap([i for i in range(len(list(out_shape)))])
                    ),
                    ir.AffineMapAttr.get(generic_map.get_submap([1])),
                    ir.AffineMapAttr.get(
                        generic_map.get_submap([i for i in range(len(list(out_shape)))])
                    ),
                ]
            ),
            ir.ArrayAttr.get(loop_type),
        )
        block = ir.Block.create_at_start(
            op.region,
            [
                result_element_type,
                ir.RankedTensorType(bias_tensor.type).element_type,
                result_element_type,
            ],
        )
        add_op = arith.AddFOp(block.arguments[1], block.arguments[0])
        block.append(add_op)
        block.append(linalg.YieldOp([add_op.result]))

    return op


def silu_op(node: SiluOp, symbol_table):
    """Preserve x / (1 + exp(-x)) rounding before activation quantization."""
    value = symbol_table[(str(node.args[0]), 0)]
    if has_nhwc_layout(value):
        value = _nhwc_to_nchw(value)
    result_type = ir.RankedTensorType(value.type)
    shape = list(result_type.shape)
    dtype = result_type.element_type
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(shape)))
    empty = tensor.EmptyOp(shape, dtype)
    op = linalg.GenericOp(
        [result_type],
        [value],
        [empty.result],
        ir.ArrayAttr.get([identity, identity]),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(shape)
        ),
    )
    block = ir.Block.create_at_start(op.region, [dtype, dtype])
    with ir.InsertionPoint(block):
        argument = block.arguments[0]
        opmath = (
            ir.F32Type.get()
            if dtype in (ir.F16Type.get(), ir.BF16Type.get())
            else dtype
        )
        if opmath != dtype:
            argument = arith.ExtFOp(opmath, argument).result
        negative = arith.NegFOp(argument).result
        exponential = (
            exp_f32(negative)
            if opmath == ir.F32Type.get()
            else math.ExpOp(negative).result
        )
        one = arith.ConstantOp(opmath, ir.FloatAttr.get(opmath, 1.0)).result
        denominator = arith.AddFOp(one, exponential).result
        result = arith.DivFOp(argument, denominator).result
        if opmath != dtype:
            result = arith.TruncFOp(dtype, result).result
        linalg.YieldOp([result])
    return op


def mean_op(node: MeanOp, symbol_table):
    """
    Import the buddy MeanOp.
    From Buddy MeanOp to MLIR TOSA operation.
    """

    def _inner_op(result_type, input1, input2):
        shift = _create_mul_shift_operand()
        return tosa.MulOp(result_type, input1, input2, shift)

    input_tensor = symbol_table.get((str(node.args[0]), 0))
    keepdim = node.args[2]
    dims = [int(x) for x in node.args[1]]
    if isinstance(dims, int):
        dims = [dims]

    act = [int(x) for x in ir.RankedTensorType(input_tensor.type).shape]
    for i, d in enumerate(dims):
        if d < 0:
            dims[i] = d + len(act)

    # PyTorch dims are NCHW; remap when SSA stuck in NHWC.
    out_meta = [int(x) for x in node.tensor_meta["shape"]]
    if len(out_meta) == 4 or len(out_meta) == 2:
        c_meta = out_meta[1]
    elif len(out_meta) == 1:
        c_meta = out_meta[0]
    else:
        c_meta = None
    if c_meta is not None and len(act) == 4 and act[3] == c_meta and act[1] != c_meta:
        dims = [{0: 0, 1: 3, 2: 1, 3: 2}[d] for d in dims]
    elif c_meta is not None and len(act) == 4 and act[1] != c_meta:
        raise RuntimeError(f"mean layout mismatch: ssa={act} out_meta={out_meta}")

    reduce_sum_result = input_tensor
    for dim_item in dims:
        reduce_dim_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), dim_item)
        reduce_sum_op = tosa.ReduceSumOp(reduce_sum_result, reduce_dim_attr)
        reduce_sum_result = reduce_sum_op.results[0]

    tensor_shp = ir.RankedTensorType(input_tensor.type).shape
    dim_size = 1

    for dim_item in dims:
        dim_size *= tensor_shp[dim_item]

    denominator_const_op = tosa.ConstOp(
        ir.DenseElementsAttr.get(memoryview(array.array("f", [dim_size])))
    )
    reciprocal_op = tosa.ReciprocalOp(
        denominator_const_op.results[0].type, denominator_const_op
    )
    ret = _gen_arith_binary_op(
        reciprocal_op.results[0], reduce_sum_op.results[0], _inner_op
    )

    if not keepdim:
        result_shp = ir.RankedTensorType(ret.results[0].type).shape
        result_shp = [siz for siz in result_shp if siz != 1]
        reshape_operand = _create_shape_operand(result_shp)
        ret = tosa.ReshapeOp(ret.results[0], reshape_operand)

    return ret


def alias_op(node: AliasOp, symbol_table):
    """
    Import the alias operation.
    From buddy graph ir's `AliasOp` operator to MLIR TOSA operations.
    Creates an alias (view) of the input tensor.
    aten.alias.default(input) -> Tensor
    """
    input1 = symbol_table.get((str(node.args[0]), 0))

    input_shape = list(ir.RankedTensorType(input1.type).shape)
    input_dtype = ir.RankedTensorType(input1.type).element_type

    output_shape = list(node.tensor_meta["shape"])
    output_dtype = mlir_element_type_get(node.tensor_meta["dtype"])

    if input_shape == output_shape and input_dtype == output_dtype:
        return input1

    # Sticky NHWC: alias must not invent NCHW type via identity.
    if len(input_shape) == 4 and len(output_shape) == 4:
        ns = [int(x) for x in input_shape]
        nw = [int(x) for x in output_shape]
        if ns == [nw[0], nw[2], nw[3], nw[1]]:
            return input1
        if nw == [ns[0], ns[2], ns[3], ns[1]]:
            out_ty = ir.RankedTensorType.get(nw, output_dtype)
            return tosa.TransposeOp(
                out_ty, input1, _create_permutation_attr([0, 2, 3, 1])
            ).result

    raise RuntimeError(
        f"alias shape/layout mismatch: ssa={input_shape} meta={output_shape}"
    )


def _bn_channel_and_broadcast(input_tensor, channel_param):
    shape = [int(x) for x in ir.RankedTensorType(input_tensor.type).shape]
    c = int(list(ir.RankedTensorType(channel_param.type).shape)[0])
    if len(shape) == 4:
        if has_nhwc_layout(input_tensor):
            return c, [1, 1, 1, c]
        if shape[1] == c:
            return c, [1, c, 1, 1]
        if shape[3] == c:
            return c, [1, 1, 1, c]
        raise RuntimeError(f"BN channel {c} mismatch with 4D shape {shape}")
    if len(shape) >= 2:
        if shape[1] == c:
            return c, [1, c] + [1] * (len(shape) - 2)
        if shape[-1] == c:
            return c, [1] * (len(shape) - 1) + [c]
    raise RuntimeError(f"BN channel {c} mismatch with shape {shape}")


def native_batch_norm_legit_op(node, symbol_table):
    """
    Import the native batch norm legit operation.
    From buddy graph ir's `NativeBatchNormLegitOp` operator to MLIR TOSA operations.
    aten._native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps)
        -> (Tensor, Tensor, Tensor)

    Returns a tuple of (output, save_mean, save_invstd) where:
    - output: normalized tensor with same shape as input
    - save_mean: mean per channel, shape (C,)
    - save_invstd: inverse std per channel, shape (C,)

    Batch normalization formula:
    output = (input - mean) / sqrt(var + eps) * weight + bias

    During training: uses batch statistics
    During inference: uses running_mean and running_var
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    weight = (
        symbol_table.get((str(node.args[1]), 0)) if node.args[1] is not None else None
    )
    bias = (
        symbol_table.get((str(node.args[2]), 0)) if node.args[2] is not None else None
    )
    running_mean = (
        symbol_table.get((str(node.args[3]), 0)) if node.args[3] is not None else None
    )
    running_var = (
        symbol_table.get((str(node.args[4]), 0)) if node.args[4] is not None else None
    )
    training = node.args[5]
    momentum = node.args[6]
    eps = node.args[7]

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    channel_ref = running_mean or weight or bias
    if channel_ref is None:
        raise RuntimeError("batch norm needs running_mean, weight, or bias")
    C, broadcast_shape = _bn_channel_and_broadcast(input_tensor, channel_ref)
    N = input_shape[0]
    spatial_dims = input_shape[2:]
    spatial_size = 1
    for d in spatial_dims:
        spatial_size *= d

    if not training and running_mean is not None and running_var is not None:
        # Inference mode: use running statistics
        mean = running_mean
        var = running_var

        scalar_broadcast_shape = [1] * len(input_shape)  # for scalars like eps

        mean_broadcast = tosa.ReshapeOp(
            mean, _create_shape_operand(broadcast_shape)
        ).result
        var_broadcast = tosa.ReshapeOp(
            var, _create_shape_operand(broadcast_shape)
        ).result

        # (input - mean)
        centered = tosa.SubOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            input_tensor,
            mean_broadcast,
        ).result

        # var + eps - reshape eps to same rank for TOSA compatibility
        eps_attr = ir.DenseElementsAttr.get(
            memoryview(array.array("f", [eps])),
            type=ir.RankedTensorType.get([1], input_dtype),
        )
        eps_tensor = tosa.ConstOp(eps_attr).result
        eps_broadcast = tosa.ReshapeOp(
            eps_tensor, _create_shape_operand(scalar_broadcast_shape)
        ).result

        var_plus_eps = tosa.AddOp(
            ir.RankedTensorType.get(broadcast_shape, input_dtype),
            var_broadcast,
            eps_broadcast,
        ).result

        # rsqrt(var + eps)
        invstd = tosa.RsqrtOp(
            ir.RankedTensorType.get(broadcast_shape, input_dtype), var_plus_eps
        ).result

        # normalized = (input - mean) * invstd
        normalized = tosa.MulOp(
            ir.RankedTensorType.get(input_shape, input_dtype), centered, invstd
        ).result

        output = normalized

        # Apply weight (gamma)
        if weight is not None:
            weight_broadcast = tosa.ReshapeOp(
                weight, _create_shape_operand(broadcast_shape)
            ).result
            output = tosa.MulOp(
                ir.RankedTensorType.get(input_shape, input_dtype),
                output,
                weight_broadcast,
            ).result

        # Apply bias (beta)
        if bias is not None:
            bias_broadcast = tosa.ReshapeOp(
                bias, _create_shape_operand(broadcast_shape)
            ).result
            output = tosa.AddOp(
                ir.RankedTensorType.get(input_shape, input_dtype),
                output,
                bias_broadcast,
            ).result

        # save_mean and save_invstd with shape (C,)
        save_mean = tosa.ReshapeOp(mean_broadcast, _create_shape_operand([C])).result
        save_invstd = tosa.ReshapeOp(invstd, _create_shape_operand([C])).result

        return output, save_mean, save_invstd

    # Training mode: compute batch statistics
    train_input = input_tensor
    train_shape = list(input_shape)
    if len(train_shape) == 4 and train_shape[3] == C:
        nchw_shape = [train_shape[0], C, train_shape[1], train_shape[2]]
        train_input = tosa.TransposeOp(
            ir.RankedTensorType.get(nchw_shape, input_dtype),
            input_tensor,
            _create_permutation_attr([0, 3, 1, 2]),
        ).result
        train_shape = nchw_shape
    elif not (len(train_shape) >= 2 and train_shape[1] == C):
        raise RuntimeError(f"BN training layout mismatch: shape={input_shape} C={C}")
    N = train_shape[0]
    spatial_size = 1
    for d in train_shape[2:]:
        spatial_size *= d
    reshaped_input = tosa.ReshapeOp(
        train_input, _create_shape_operand([N, C, spatial_size])
    ).result

    # Compute mean: reduce over N and spatial dimensions
    # First reduce over spatial (axis=2)
    axis_attr_2 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2)
    spatial_sum = tosa.ReduceSumOp(reshaped_input, axis_attr_2).results[0]  # (N, C, 1)

    # Then reduce over batch (axis=0)
    axis_attr_0 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
    batch_spatial_sum = tosa.ReduceSumOp(spatial_sum, axis_attr_0).results[
        0
    ]  # (1, C, 1)

    # mean = sum / (N * spatial_size) - reshape divisor for TOSA compatibility
    divisor = N * spatial_size
    divisor_attr = ir.DenseElementsAttr.get(
        memoryview(array.array("f", [1.0 / divisor])),
        type=ir.RankedTensorType.get([1], input_dtype),
    )
    divisor_tensor = tosa.ConstOp(divisor_attr).result
    divisor_3d = tosa.ReshapeOp(divisor_tensor, _create_shape_operand([1, 1, 1])).result

    mean_3d = tosa.MulOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype),
        batch_spatial_sum,
        divisor_3d,
    ).result

    # (x - mean)
    diff = tosa.SubOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype),
        reshaped_input,
        mean_3d,
    ).result

    # (x - mean)^2
    squared = tosa.MulOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype), diff, diff
    ).result

    # sum of squared
    sq_spatial_sum = tosa.ReduceSumOp(squared, axis_attr_2).results[0]  # (N, C, 1)
    sq_batch_sum = tosa.ReduceSumOp(sq_spatial_sum, axis_attr_0).results[0]  # (1, C, 1)

    # variance = sum / (N * spatial_size)
    var_3d = tosa.MulOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype),
        sq_batch_sum,
        divisor_3d,
    ).result

    # var + eps - reshape eps for TOSA compatibility
    eps_attr = ir.DenseElementsAttr.get(
        memoryview(array.array("f", [eps])),
        type=ir.RankedTensorType.get([1], input_dtype),
    )
    eps_tensor = tosa.ConstOp(eps_attr).result
    eps_3d = tosa.ReshapeOp(eps_tensor, divisor_3d_shape_operand).result

    var_plus_eps = tosa.AddOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype), var_3d, eps_3d
    ).result

    # invstd = rsqrt(var + eps)
    invstd_3d = tosa.RsqrtOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype), var_plus_eps
    ).result

    # normalized = (x - mean) * invstd
    normalized_3d = tosa.MulOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype),
        diff,
        invstd_3d,
        shift,
    ).result

    # Reshape back to original shape
    train_shape_operand = _create_shape_operand(train_shape)
    output = tosa.ReshapeOp(normalized_3d, train_shape_operand).result
    if list(input_shape) != train_shape:
        output = tosa.TransposeOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            _create_permutation_attr([0, 2, 3, 1]),
        ).result

    # Apply weight (gamma)
    if weight is not None:
        broadcast_shape_operand = _create_shape_operand(broadcast_shape)
        weight_broadcast = tosa.ReshapeOp(weight, broadcast_shape_operand).result
        output = tosa.MulOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            weight_broadcast,
            shift,
        ).result

    # Apply bias (beta)
    if bias is not None:
        broadcast_shape_operand = _create_shape_operand(broadcast_shape)
        bias_broadcast = tosa.ReshapeOp(bias, broadcast_shape_operand).result
        output = tosa.AddOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            bias_broadcast,
        ).result

    # Return (output, save_mean, save_invstd)
    c_shape_operand = _create_shape_operand([C])
    save_mean = tosa.ReshapeOp(mean_3d, c_shape_operand).result
    save_invstd = tosa.ReshapeOp(invstd_3d, c_shape_operand).result

    return output, save_mean, save_invstd


def native_batch_norm_legit_no_stats_op(node, symbol_table):
    """
    Import the native batch norm legit no_stats operation.
    From buddy graph ir's `NativeBatchNormLegitNoStatsOp` operator to MLIR TOSA operations.
    aten._native_batch_norm_legit.no_stats(input, weight, bias, training, momentum, eps)
        -> (Tensor, Tensor, Tensor)

    This variant doesn't use running statistics, always computes from batch.
    Returns: (output, save_mean, save_invstd)
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    weight = (
        symbol_table.get((str(node.args[1]), 0)) if node.args[1] is not None else None
    )
    bias = (
        symbol_table.get((str(node.args[2]), 0)) if node.args[2] is not None else None
    )
    training = node.args[3]
    momentum = node.args[4]
    eps = node.args[5]

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    channel_ref = weight or bias
    if channel_ref is None:
        raise RuntimeError("batch norm no_stats needs weight or bias")
    C, broadcast_shape = _bn_channel_and_broadcast(input_tensor, channel_ref)

    train_input = input_tensor
    train_shape = list(input_shape)
    if len(train_shape) == 4 and train_shape[3] == C:
        nchw_shape = [train_shape[0], C, train_shape[1], train_shape[2]]
        train_input = tosa.TransposeOp(
            ir.RankedTensorType.get(nchw_shape, input_dtype),
            input_tensor,
            _create_permutation_attr([0, 3, 1, 2]),
        ).result
        train_shape = nchw_shape
    elif not (len(train_shape) >= 2 and train_shape[1] == C):
        raise RuntimeError(f"BN no_stats layout mismatch: shape={input_shape} C={C}")
    N = train_shape[0]
    spatial_size = 1
    for d in train_shape[2:]:
        spatial_size *= d

    reshaped_shape_operand = _create_shape_operand([N, C, spatial_size])
    reshaped_input = tosa.ReshapeOp(train_input, reshaped_shape_operand).result

    # Compute mean: reduce over N and spatial dimensions
    axis_attr_2 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2)
    spatial_sum = tosa.ReduceSumOp(reshaped_input, axis_attr_2).results[0]  # (N, C, 1)

    axis_attr_0 = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
    batch_spatial_sum = tosa.ReduceSumOp(spatial_sum, axis_attr_0).results[
        0
    ]  # (1, C, 1)

    # mean = sum / (N * spatial_size)
    divisor = N * spatial_size
    divisor_attr = ir.DenseElementsAttr.get(
        memoryview(array.array("f", [1.0 / divisor])),
        type=ir.RankedTensorType.get([1], input_dtype),
    )
    divisor_tensor = tosa.ConstOp(divisor_attr).result
    divisor_3d_shape_operand = _create_shape_operand([1, 1, 1])
    divisor_3d = tosa.ReshapeOp(divisor_tensor, divisor_3d_shape_operand).result

    shift = _create_mul_shift_operand()
    mean_3d = tosa.MulOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype),
        batch_spatial_sum,
        divisor_3d,
        shift,
    ).result

    # (x - mean)
    diff = tosa.SubOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype),
        reshaped_input,
        mean_3d,
    ).result

    # (x - mean)^2
    squared = tosa.MulOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype),
        diff,
        diff,
        shift,
    ).result

    # sum of squared
    sq_spatial_sum = tosa.ReduceSumOp(squared, axis_attr_2).results[0]  # (N, C, 1)
    sq_batch_sum = tosa.ReduceSumOp(sq_spatial_sum, axis_attr_0).results[0]  # (1, C, 1)

    # variance = sum / (N * spatial_size)
    var_3d = tosa.MulOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype),
        sq_batch_sum,
        divisor_3d,
        shift,
    ).result

    # var + eps - reshape eps to match [1, C, 1] shape for TOSA compatibility
    eps_attr = ir.DenseElementsAttr.get(
        memoryview(array.array("f", [eps])),
        type=ir.RankedTensorType.get([1], input_dtype),
    )
    eps_tensor = tosa.ConstOp(eps_attr).result
    eps_3d = tosa.ReshapeOp(eps_tensor, divisor_3d_shape_operand).result

    var_plus_eps = tosa.AddOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype), var_3d, eps_3d
    ).result

    # invstd = rsqrt(var + eps)
    invstd_3d = tosa.RsqrtOp(
        ir.RankedTensorType.get([1, C, 1], input_dtype), var_plus_eps
    ).result

    # normalized = (x - mean) * invstd
    normalized_3d = tosa.MulOp(
        ir.RankedTensorType.get([N, C, spatial_size], input_dtype),
        diff,
        invstd_3d,
        shift,
    ).result

    # Reshape back to original shape
    train_shape_operand = _create_shape_operand(train_shape)
    output = tosa.ReshapeOp(normalized_3d, train_shape_operand).result
    if list(input_shape) != train_shape:
        output = tosa.TransposeOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            _create_permutation_attr([0, 2, 3, 1]),
        ).result

    # Apply weight (gamma)
    if weight is not None:
        broadcast_shape_operand = _create_shape_operand(broadcast_shape)
        weight_broadcast = tosa.ReshapeOp(weight, broadcast_shape_operand).result
        output = tosa.MulOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            weight_broadcast,
            shift,
        ).result

    # Apply bias (beta)
    if bias is not None:
        broadcast_shape_operand = _create_shape_operand(broadcast_shape)
        bias_broadcast = tosa.ReshapeOp(bias, broadcast_shape_operand).result
        output = tosa.AddOp(
            ir.RankedTensorType.get(input_shape, input_dtype),
            output,
            bias_broadcast,
        ).result

    # Return (output, save_mean, save_invstd)
    c_shape_operand = _create_shape_operand([C])
    save_mean = tosa.ReshapeOp(mean_3d, c_shape_operand).result
    save_invstd = tosa.ReshapeOp(invstd_3d, c_shape_operand).result

    return output, save_mean, save_invstd


def native_batch_norm_legit_no_training_op(node, symbol_table):
    """
    Import the native batch norm legit no_training operation.
    From buddy graph ir's `NativeBatchNormLegitNoTrainingOp` operator to MLIR TOSA operations.
    aten._native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)
        -> (Tensor, Tensor, Tensor)

    This variant is inference only, always uses running statistics.
    Returns: (output, save_mean, save_invstd)
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    weight = (
        symbol_table.get((str(node.args[1]), 0)) if node.args[1] is not None else None
    )
    bias = (
        symbol_table.get((str(node.args[2]), 0)) if node.args[2] is not None else None
    )
    running_mean = symbol_table.get((str(node.args[3]), 0))
    running_var = symbol_table.get((str(node.args[4]), 0))
    momentum = node.args[5]
    eps = node.args[6]

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    channel_ref = running_mean or weight or bias
    if channel_ref is None:
        raise RuntimeError("batch norm needs running_mean, weight, or bias")
    C, broadcast_shape = _bn_channel_and_broadcast(input_tensor, channel_ref)

    # Inference mode: use running statistics
    scalar_broadcast_shape = [1] * len(input_shape)  # for scalars like eps

    broadcast_shape_operand = _create_shape_operand(broadcast_shape)
    mean_broadcast = tosa.ReshapeOp(running_mean, broadcast_shape_operand).result
    var_broadcast = tosa.ReshapeOp(running_var, broadcast_shape_operand).result

    # var + eps - reshape eps to same rank for TOSA compatibility
    eps_attr = ir.DenseElementsAttr.get(
        memoryview(array.array("f", [eps])),
        type=ir.RankedTensorType.get([1], input_dtype),
    )
    eps_tensor = tosa.ConstOp(eps_attr).result
    scalar_broadcast_shape_operand = _create_shape_operand(scalar_broadcast_shape)
    eps_broadcast = tosa.ReshapeOp(eps_tensor, scalar_broadcast_shape_operand).result

    var_plus_eps = tosa.AddOp(
        ir.RankedTensorType.get(broadcast_shape, input_dtype),
        var_broadcast,
        eps_broadcast,
    ).result

    # PyTorch computes 1 / sqrt(var + eps). An approximate reciprocal
    # square root can cross an activation quantization threshold on NVIDIA.
    rank = len(input_shape)
    identity = ir.AffineMap.get_identity(rank)
    channel_type = ir.RankedTensorType.get(broadcast_shape, input_dtype)
    inverse_empty = tensor.EmptyOp(broadcast_shape, input_dtype)
    inverse = linalg.GenericOp(
        [channel_type],
        [var_plus_eps],
        [inverse_empty.result],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(identity)] * 2),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        ),
    )
    inverse_block = ir.Block.create_at_start(inverse.region, [input_dtype] * 2)
    with ir.InsertionPoint(inverse_block):
        root = math.SqrtOp(inverse_block.arguments[0]).result
        one = arith.ConstantOp(input_dtype, ir.FloatAttr.get(input_dtype, 1.0)).result
        reciprocal = arith.DivFOp(one, root).result
        linalg.YieldOp([reciprocal])
    invstd = inverse.result

    # Match PyTorch inference: alpha = invstd * gamma,
    # beta = bias - mean * alpha, output = input * alpha + beta.
    # Reassociating this into (input - mean) * invstd * gamma + bias
    # changes FP32 rounding before downstream activation quantization.
    alpha = invstd
    if weight is not None:
        gamma = tosa.ReshapeOp(weight, broadcast_shape_operand).result
        alpha = _gen_arith_binary_op(invstd, gamma, tosa.MulOp).result
    if bias is None:
        mean_scaled = _gen_arith_binary_op(mean_broadcast, alpha, tosa.MulOp).result
        beta = tosa.NegateOp(
            mean_scaled.type,
            mean_scaled,
            _create_zero_point_tensor(mean_scaled),
            _create_zero_point_tensor(mean_scaled),
        ).result
    else:
        bias_broadcast = tosa.ReshapeOp(bias, broadcast_shape_operand).result
        beta_empty = tensor.EmptyOp(broadcast_shape, input_dtype)
        beta_op = linalg.GenericOp(
            [channel_type],
            [mean_broadcast, alpha, bias_broadcast],
            [beta_empty.result],
            ir.ArrayAttr.get([ir.AffineMapAttr.get(identity)] * 4),
            ir.ArrayAttr.get(
                [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
            ),
        )
        beta_block = ir.Block.create_at_start(beta_op.region, [input_dtype] * 4)
        with ir.InsertionPoint(beta_block):
            negative_mean = arith.NegFOp(beta_block.arguments[0]).result
            coefficient = math.FmaOp(
                negative_mean, beta_block.arguments[1], beta_block.arguments[2]
            ).result
            linalg.YieldOp([coefficient])
        beta = beta_op.result
    # Keep the final multiply-add fused: rounding the product separately
    # changes values close to downstream INT8 rounding thresholds.
    channel_map = ir.AffineMap.get(
        rank,
        0,
        [
            (ir.AffineExpr.get_dim(i) if extent != 1 else ir.AffineExpr.get_constant(0))
            for i, extent in enumerate(broadcast_shape)
        ],
    )
    result_type = ir.RankedTensorType.get(input_shape, input_dtype)
    empty = tensor.EmptyOp(input_shape, input_dtype)
    fused = linalg.GenericOp(
        [result_type],
        [input_tensor, alpha, beta],
        [empty.result],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(m)
                for m in (identity, channel_map, channel_map, identity)
            ]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank
        ),
    )
    block = ir.Block.create_at_start(fused.region, [input_dtype] * 4)
    with ir.InsertionPoint(block):
        value = math.FmaOp(*block.arguments[:3]).result
        linalg.YieldOp([value])
    if has_nhwc_layout(input_tensor):
        fused.operation.attributes["buddy.layout"] = ir.StringAttr.get("nhwc")
    output = fused.result

    # save_mean and save_invstd with shape (C,)
    c_shape_operand = _create_shape_operand([C])
    save_mean = tosa.ReshapeOp(mean_broadcast, c_shape_operand).result
    save_invstd = tosa.ReshapeOp(invstd, c_shape_operand).result

    return output, save_mean, save_invstd


def upsample_bilinear2d_vec_op(node, symbol_table):
    """
    Import the upsample_bilinear2d.vec operation.
    From buddy graph ir's `UpsampleBilinear2dVecOp` operator to MLIR TOSA operations.
    aten.upsample_bilinear2d.vec(input, output_size, align_corners, scale_factors) -> Tensor

    Uses TOSA resize operation with BILINEAR mode.
    Input: NCHW format, needs to be converted to NHWC for TOSA.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    output_size = node.args[1]  # [H_out, W_out] or None
    align_corners = node.args[2]
    scale_factors = (
        node.args[3] if len(node.args) > 3 else None
    )  # [scale_h, scale_w] or None

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    channels = int(list(node.tensor_meta["shape"])[1])
    src = node.args[0]
    nchw_meta = (
        list(src.tensor_meta["shape"])
        if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
        else None
    )
    input_tensor, N, H_in, W_in, C = _to_nhwc(
        input_tensor, channels, input_dtype, nchw_meta
    )

    # Determine output size
    if output_size is not None:
        H_out, W_out = output_size
    elif scale_factors is not None:
        H_out = int(H_in * scale_factors[0])
        W_out = int(W_in * scale_factors[1])
    else:
        raise ValueError("Either output_size or scale_factors must be provided")

    # Calculate scale for TOSA resize
    # TOSA scale format: [scale_y_n, scale_y_d, scale_x_n, scale_x_d]
    # For exact output size: scale = output_size / input_size
    scale_y_n = H_out
    scale_y_d = H_in
    scale_x_n = W_out
    scale_x_d = W_in

    scale_attr = ir._denseI64ArrayAttr(
        [scale_y_n, scale_y_d, scale_x_n, scale_x_d], None
    )
    offset_attr = ir._denseI64ArrayAttr([0, 0], None)
    border_attr = ir._denseI64ArrayAttr([0, 0], None)
    mode_attr = ir.StringAttr.get("BILINEAR")

    # Apply resize in NHWC format; keep NHWC output.
    nhwc_out_type = ir.RankedTensorType.get([N, H_out, W_out, C], input_dtype)
    resized = tosa.ResizeOp(
        nhwc_out_type,
        input_tensor,
        scale_attr,
        offset_attr,
        border_attr,
        mode_attr,
    )

    return resized


def upsample_nearest2d_vec_op(node, symbol_table):
    """
    Import the upsample_nearest2d.vec operation.
    From buddy graph ir's `UpsampleNearest2dVecOp` operator to MLIR TOSA operations.
    aten.upsample_nearest2d.vec(input, output_size, scale_factors) -> Tensor

    Uses TOSA resize operation with NEAREST mode.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    output_size = node.args[1]  # [H_out, W_out] or None
    scale_factors = (
        node.args[2] if len(node.args) > 2 else None
    )  # [scale_h, scale_w] or None

    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    channels = int(list(node.tensor_meta["shape"])[1])
    src = node.args[0]
    nchw_meta = (
        list(src.tensor_meta["shape"])
        if hasattr(src, "tensor_meta") and "shape" in src.tensor_meta
        else None
    )
    input_tensor, N, H_in, W_in, C = _to_nhwc(
        input_tensor, channels, input_dtype, nchw_meta
    )

    # Determine output size
    if output_size is not None:
        H_out, W_out = output_size
    elif scale_factors is not None:
        H_out = int(H_in * scale_factors[0])
        W_out = int(W_in * scale_factors[1])
    else:
        raise ValueError("Either output_size or scale_factors must be provided")

    scale_attr = ir._denseI64ArrayAttr([H_out, H_in, W_out, W_in], None)
    offset_attr = ir._denseI64ArrayAttr([0, 0], None)
    border_attr = ir._denseI64ArrayAttr([0, 0], None)
    mode_attr = ir.StringAttr.get("NEAREST")

    nhwc_out_type = ir.RankedTensorType.get([N, H_out, W_out, C], input_dtype)
    return tosa.ResizeOp(
        nhwc_out_type,
        input_tensor,
        scale_attr,
        offset_attr,
        border_attr,
        mode_attr,
    )


from buddy.compiler.ops.tosa import convert_element_type_op, where_op
from buddy.compiler.ops.tosa import ops_registry as upstream_registry
from .linalg import ops_registry as linalg_registry

ops_registry = {
    **linalg_registry,
    **upstream_registry,
    "WhereOp": where_op,
    "ToCopyOp": convert_element_type_op,
    "AddOp": add_op,
    "MulOp": mul_op,
    "SubOp": sub_op,
    "AmaxOp": amax_op,
    "DivOp": div_op,
    "ReshapeOp": reshape_op,
    "ViewOp": reshape_op,
    "MaxPool2dOp": maxpool2d_op,
    "AvgPool2dOp": avg_pool2d_op,
    "AdaptiveAvgPool2dOp": adaptive_avg_pool2d_op,
    "Conv2dOp": convolution2d_op,
    "Int8Conv2dOp": int8_conv2d_op,
    "Int8MatmulOp": int8_matmul_op,
    "Int8QuantizeOp": int8_quantize_op,
    "SiluOp": silu_op,
    "MeanOp": mean_op,
    "AliasOp": alias_op,
    "NativeBatchNormLegitOp": native_batch_norm_legit_op,
    "NativeBatchNormLegitNoStatsOp": native_batch_norm_legit_no_stats_op,
    "NativeBatchNormLegitNoTrainingOp": native_batch_norm_legit_no_training_op,
    "UpsampleBilinear2dVecOp": upsample_bilinear2d_vec_op,
    "UpsampleNearest2dVecOp": upsample_nearest2d_vec_op,
}
