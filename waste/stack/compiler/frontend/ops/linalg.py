# ===- linalg.py ---------------------------------------------------------------
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
# The registry of mappings from Buddy Graph to MLIR linalg dialect operations.
#
# ===---------------------------------------------------------------------------

import array
import copy
import buddy_mlir.ir as ir
import numpy
from buddy_mlir.dialects import (
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
from buddy.compiler.graph import *
from buddy.compiler.graph.graph import TensorDType
from buddy.compiler.graph.transform.trace.lowering import _trace_attrs, has_trace
from buddy.compiler.ops.utils import *
from stack.compiler.frontend.operation import *
from buddy.compiler.ops.linalg import (
    _as_value,
    _const_shape_operand,
    _max_pool_indices_to_offsets_3d,
    _normalize_pool_param,
    max_pool3d_op,
)


def _perm_attr(perm: list[int]) -> ir.Attribute:
    return ir.DenseI32ArrayAttr.get([int(x) for x in perm])


def _nhwc_to_nchw(value: ir.Value) -> ir.Value:
    shape = [int(x) for x in ir.RankedTensorType(value.type).shape]
    if len(shape) != 4:
        raise RuntimeError(f"_nhwc_to_nchw expects rank-4, got {shape}")
    n, h, w, c = shape
    elem = ir.RankedTensorType(value.type).element_type
    out_ty = ir.RankedTensorType.get([n, c, h, w], elem)
    return tosa.TransposeOp(out_ty, value, _perm_attr([0, 3, 1, 2])).result


def _nchw_to_nhwc(value: ir.Value) -> ir.Value:
    shape = [int(x) for x in ir.RankedTensorType(value.type).shape]
    if len(shape) != 4:
        raise RuntimeError(f"_nchw_to_nhwc expects rank-4, got {shape}")
    n, c, h, w = shape
    elem = ir.RankedTensorType(value.type).element_type
    out_ty = ir.RankedTensorType.get([n, h, w, c], elem)
    return tosa.TransposeOp(out_ty, value, _perm_attr([0, 2, 3, 1])).result


def _split_via_tosa_slice(
    input_tensor: ir.Value,
    input_shape: list[int],
    dim: int,
    split_sizes: list[int],
) -> list[ir.Value]:
    # tosa.slice / extract_slice on non-dim1 (esp. NHWC C) lowers wrong;
    # always slice on dim1 via transpose.
    rank = len(input_shape)
    if dim != 1:
        perm = list(range(rank))
        perm[1], perm[dim] = perm[dim], perm[1]
        inv = [0] * rank
        for i, j in enumerate(perm):
            inv[j] = i
        elem = ir.RankedTensorType(input_tensor.type).element_type
        t_shape = [input_shape[i] for i in perm]
        t = tosa.TransposeOp(
            ir.RankedTensorType.get(t_shape, elem),
            input_tensor,
            _perm_attr(perm),
        ).result
        parts = _split_via_tosa_slice(t, t_shape, 1, split_sizes)
        out = []
        for p in parts:
            ps = [int(x) for x in ir.RankedTensorType(p.type).shape]
            o_shape = [ps[i] for i in inv]
            out.append(
                tosa.TransposeOp(
                    ir.RankedTensorType.get(o_shape, elem),
                    p,
                    _perm_attr(inv),
                ).result
            )
        return out

    elem = ir.RankedTensorType(input_tensor.type).element_type
    results = []
    offset = 0
    for size in split_sizes:
        start = [0] * rank
        start[1] = offset
        out_shape = list(input_shape)
        out_shape[1] = size
        out_ty = ir.RankedTensorType.get(out_shape, elem)
        results.append(
            tosa.SliceOp(
                out_ty,
                input_tensor,
                _const_shape_operand(start),
                _const_shape_operand(out_shape),
            ).result
        )
        offset += size
    return results


def cat_op(
    node: CatOp,
    symbol_table: dict[tuple[str, int], ir.Operation],
):
    """
    Import the tensor concate operation.
    From buddy CatOp to MLIR tensor `insert_slice`
    operation.

    Note: This op, concate multiple input tensors.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the tensor.insert_slice op.
    """
    # Support both 2 args and variable number of inputs
    inputs = node.args[0]  # List of input tensors
    # dim is optional, default to 0 if not provided
    dim = int(node.args[1]) if len(node.args) > 1 else 0

    # Get all input tensors
    input_tensors = []
    input_shapes = []

    for inp in inputs:
        t = symbol_table.get((str(inp), 0))
        if t is None:
            raise ValueError(f"missing cat input for {node.name}: {inp}")
        input_tensors.append(t)
        input_shapes.append(list(ir.RankedTensorType(t.type).shape))

    if len(input_tensors) == 0:
        raise ValueError(f"cat has no inputs: {node.name}")

    rank = len(input_shapes[0])
    if dim < 0:
        dim += rank

    # Sticky NHWC: PyTorch cat dim is NCHW; remap when SSA is NHWC.
    out_meta = [int(x) for x in node.tensor_meta["shape"]]
    if rank == 4 and dim == 1 and len(out_meta) == 4:
        n, out_c, h, w = out_meta
        nchw_inputs = []
        channels = 0
        for value, shape in zip(input_tensors, input_shapes):
            if shape[0] == n and shape[1] == h and shape[2] == w:
                nchw_inputs.append(_nhwc_to_nchw(value))
                channels += shape[3]
            elif shape[0] == n and shape[2] == h and shape[3] == w:
                nchw_inputs.append(value)
                channels += shape[1]
            else:
                raise RuntimeError(
                    f"channel cat input layout mismatch: shape={shape} meta={out_meta}"
                )
        if channels != out_c:
            raise RuntimeError(
                f"channel cat count mismatch: inputs={input_shapes} meta={out_meta}"
            )
        return tosa.ConcatOp(nchw_inputs, 1).result

    if (
        rank == 4
        and len(out_meta) == 4
        and input_shapes[0][1] == out_meta[2]
        and input_shapes[0][2] == out_meta[3]
        and sum(s[3] for s in input_shapes) == out_meta[1]
    ):
        dim = {0: 0, 1: 3, 2: 1, 3: 2}[dim]

    for s in input_shapes[1:]:
        if len(s) != rank:
            raise RuntimeError(f"cat rank mismatch: {input_shapes[0]} vs {s}")
        for d in range(rank):
            if d != dim and s[d] != input_shapes[0][d]:
                raise RuntimeError(
                    f"cat non-concat dim{d} mismatch: shapes={input_shapes} "
                    f"dim={dim} meta={out_meta}"
                )

    output_shape = list(input_shapes[0])
    output_shape[dim] = sum(s[dim] for s in input_shapes)

    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)

    # insert_slice on NHWC C lowers wrong; concat on NCHW axis=1 instead.
    nhwc_c = (
        rank == 4
        and dim == 3
        and input_shapes[0][1] == input_shapes[0][2]
        and input_shapes[0][3] < output_shape[3]
    )
    if nhwc_c:
        nchw_inputs = [_nhwc_to_nchw(t) for t in input_tensors]
        cat_nchw = tosa.ConcatOp(nchw_inputs, 1).result
        return cat_nchw

    output = tensor.EmptyOp(output_shape, mlir_dtype)

    offset = [0] * rank
    stride_attr = ir._denseI64ArrayAttr([1] * rank, None)

    current_result = output.result

    for input_tensor, input_shape in zip(input_tensors, input_shapes):
        offset_attr = ir._denseI64ArrayAttr(offset, None)
        size_attr = ir._denseI64ArrayAttr(input_shape, None)
        insert_op = tensor.InsertSliceOp(
            input_tensor,
            current_result,
            [],
            [],
            [],
            offset_attr,
            size_attr,
            stride_attr,
        )
        current_result = insert_op.result
        offset[dim] += input_shape[dim]

    return insert_op


def split_op(node: SplitOp, symbol_table):
    """
    Split the input tensor into smaller tensors along the specified dimension.

    Args:
        node (SplitOp): The split operation node with metadata.
        symbol_table: Mapping of variable names to tensor references.

    Returns:
        List[Tensor]: List of split tensors.
    """
    # Get the input tensor and parameters
    input_tensor = symbol_table.get((str(node.args[0]), 0), node.args[0])
    if not hasattr(input_tensor, "type"):
        raise ValueError(f"missing split input for {node.name}: {node.args[0]}")
    split_size = node.args[1]  # Size of each split tensor
    input_shape = [int(x) for x in input_tensor.type.shape]
    dim = int(node.args[2])  # Dimension to split along (PyTorch NCHW)
    if dim < 0:
        dim += len(input_shape)

    # Sticky NHWC: remap channel/spatial dims from NCHW numbering.
    meta_shapes = node.tensor_meta["shape"]
    if (
        isinstance(meta_shapes, (list, tuple))
        and len(meta_shapes) > 0
        and hasattr(meta_shapes[0], "__len__")
        and len(input_shape) == 4
        and len(meta_shapes[0]) == 4
    ):
        m0 = [int(x) for x in meta_shapes[0]]
        # Full NCHW concat along dim would match input meta; NHWC has C last.
        if (
            input_shape[3] == sum(int(s[1]) for s in meta_shapes)
            and input_shape[1] == m0[2]
        ):
            dim = {0: 0, 1: 3, 2: 1, 3: 2}[dim]

    split_count = (input_shape[dim] + split_size - 1) // split_size  # Round up
    if (
        isinstance(meta_shapes, (list, tuple))
        and len(meta_shapes) > 0
        and hasattr(meta_shapes[0], "__len__")
        and split_count != len(meta_shapes)
    ):
        raise RuntimeError(
            f"split count mismatch: ssa_dim={dim} count={split_count} "
            f"meta_outs={len(meta_shapes)} shape={input_shape}"
        )

    split_sizes = [
        min(split_size, input_shape[dim] - i * split_size) for i in range(split_count)
    ]
    return _split_via_tosa_slice(input_tensor, input_shape, dim, split_sizes)


def _infer_pool2d_nhwc(shape: list[int]) -> bool:
    if len(shape) != 4:
        raise RuntimeError(f"expected rank-4 pool tensor, got {shape}")
    _, a, b, c = shape
    if a == b and c > a:
        return True
    if b == c and a > b:
        return False
    raise RuntimeError(f"cannot infer pool layout from shape {shape}")


def _pool2d_input_hw(input_shape: list[int]) -> list[int]:
    if len(input_shape) != 4:
        raise RuntimeError(f"expected rank-4 pool input, got {input_shape}")
    _, a, b, c = input_shape
    if a == b and c > a:
        return [a, b]
    if b == c and a > b:
        return [b, c]
    raise RuntimeError(f"cannot infer pool input_hw from shape {input_shape}")


def _emit_max_pool_offset_store(
    flat_idx,
    oh,
    ow,
    out_idx,
    w_const,
    kw_const,
    sh_const,
    sw_const,
    ph_const,
    pw_const,
    dh_const,
    dw_const,
    i64,
    i8,
    offsets_memref,
):
    ih = arith.DivSIOp(flat_idx, w_const).result
    iw = arith.RemSIOp(flat_idx, w_const).result

    oh_i64 = arith.IndexCastOp(i64, oh).result
    ow_i64 = arith.IndexCastOp(i64, ow).result

    hbase = arith.SubIOp(arith.MulIOp(oh_i64, sh_const).result, ph_const).result
    wbase = arith.SubIOp(arith.MulIOp(ow_i64, sw_const).result, pw_const).result

    h_inc = arith.DivSIOp(arith.SubIOp(ih, hbase).result, dh_const).result
    w_inc = arith.DivSIOp(arith.SubIOp(iw, wbase).result, dw_const).result
    offset_i64 = arith.AddIOp(arith.MulIOp(h_inc, kw_const).result, w_inc).result
    offset_i8 = arith.TruncIOp(i8, offset_i64).result
    memref.StoreOp(offset_i8, offsets_memref.result, out_idx)


def _max_pool_indices_to_offsets_2d(
    indices_tensor: ir.Value,
    input_hw: list[int],
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
):
    indices_tensor = _as_value(indices_tensor)
    shape = [int(x) for x in ir.RankedTensorType(indices_tensor.type).shape]
    nhwc = _infer_pool2d_nhwc(shape)
    i64 = ir.IntegerType.get_signless(64)
    i8 = ir.IntegerType.get_signless(8)
    index_ty = ir.IndexType.get()

    offsets_memref_type = ir.MemRefType.get(shape, i8)
    offsets_memref = memref.AllocOp(offsets_memref_type, [], [])

    lb = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, 0)).result
    step = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, 1)).result

    n_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[0])).result
    if nhwc:
        oh_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[1])
        ).result
        ow_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[2])
        ).result
        c_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[3])).result
    else:
        c_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[1])).result
        oh_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[2])
        ).result
        ow_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[3])
        ).result

    w_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(input_hw[1]))).result
    kw_const = arith.ConstantOp(
        i64, ir.IntegerAttr.get(i64, int(kernel_size[1]))
    ).result
    sh_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(stride[0]))).result
    sw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(stride[1]))).result
    ph_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(padding[0]))).result
    pw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(padding[1]))).result
    dh_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(dilation[0]))).result
    dw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(dilation[1]))).result

    n_for = scf.ForOp(lb, n_ub, step)
    with ir.InsertionPoint(n_for.body):
        n = n_for.induction_variable
        if nhwc:
            oh_for = scf.ForOp(lb, oh_ub, step)
            with ir.InsertionPoint(oh_for.body):
                oh = oh_for.induction_variable
                ow_for = scf.ForOp(lb, ow_ub, step)
                with ir.InsertionPoint(ow_for.body):
                    ow = ow_for.induction_variable
                    c_for = scf.ForOp(lb, c_ub, step)
                    with ir.InsertionPoint(c_for.body):
                        c = c_for.induction_variable
                        flat_idx = tensor.ExtractOp(
                            indices_tensor, [n, oh, ow, c]
                        ).result
                        out_idx = [n, oh, ow, c]
                        _emit_max_pool_offset_store(
                            flat_idx,
                            oh,
                            ow,
                            out_idx,
                            w_const,
                            kw_const,
                            sh_const,
                            sw_const,
                            ph_const,
                            pw_const,
                            dh_const,
                            dw_const,
                            i64,
                            i8,
                            offsets_memref,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        else:
            c_for = scf.ForOp(lb, c_ub, step)
            with ir.InsertionPoint(c_for.body):
                c = c_for.induction_variable
                oh_for = scf.ForOp(lb, oh_ub, step)
                with ir.InsertionPoint(oh_for.body):
                    oh = oh_for.induction_variable
                    ow_for = scf.ForOp(lb, ow_ub, step)
                    with ir.InsertionPoint(ow_for.body):
                        ow = ow_for.induction_variable
                        flat_idx = tensor.ExtractOp(
                            indices_tensor, [n, c, oh, ow]
                        ).result
                        out_idx = [n, c, oh, ow]
                        _emit_max_pool_offset_store(
                            flat_idx,
                            oh,
                            ow,
                            out_idx,
                            w_const,
                            kw_const,
                            sh_const,
                            sw_const,
                            ph_const,
                            pw_const,
                            dh_const,
                            dw_const,
                            i64,
                            i8,
                            offsets_memref,
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    offsets_type = ir.RankedTensorType.get(shape, i8)
    return bufferization.ToTensorOp(
        offsets_type, offsets_memref.result, restrict=True
    ).result


def _max_pool_offsets_to_indices_2d(
    offsets_tensor: ir.Value,
    input_hw: list[int],
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
):
    offsets_tensor = _as_value(offsets_tensor)
    shape = [int(x) for x in ir.RankedTensorType(offsets_tensor.type).shape]
    nhwc = _infer_pool2d_nhwc(shape)
    i64 = ir.IntegerType.get_signless(64)
    i8 = ir.IntegerType.get_signless(8)
    index_ty = ir.IndexType.get()

    indices_memref_type = ir.MemRefType.get(shape, i64)
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    lb = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, 0)).result
    step = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, 1)).result

    n_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[0])).result
    if nhwc:
        oh_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[1])
        ).result
        ow_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[2])
        ).result
        c_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[3])).result
    else:
        c_ub = arith.ConstantOp(index_ty, ir.IntegerAttr.get(index_ty, shape[1])).result
        oh_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[2])
        ).result
        ow_ub = arith.ConstantOp(
            index_ty, ir.IntegerAttr.get(index_ty, shape[3])
        ).result

    in_w_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(input_hw[1]))).result
    kw_const = arith.ConstantOp(
        i64, ir.IntegerAttr.get(i64, int(kernel_size[1]))
    ).result
    sh_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(stride[0]))).result
    sw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(stride[1]))).result
    ph_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(padding[0]))).result
    pw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(padding[1]))).result
    dh_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(dilation[0]))).result
    dw_const = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, int(dilation[1]))).result

    n_for = scf.ForOp(lb, n_ub, step)
    with ir.InsertionPoint(n_for.body):
        n = n_for.induction_variable
        if nhwc:
            oh_for = scf.ForOp(lb, oh_ub, step)
            with ir.InsertionPoint(oh_for.body):
                oh = oh_for.induction_variable
                ow_for = scf.ForOp(lb, ow_ub, step)
                with ir.InsertionPoint(ow_for.body):
                    ow = ow_for.induction_variable
                    c_for = scf.ForOp(lb, c_ub, step)
                    with ir.InsertionPoint(c_for.body):
                        c = c_for.induction_variable
                        off_i8 = tensor.ExtractOp(offsets_tensor, [n, oh, ow, c]).result
                        off_i64 = arith.ExtSIOp(i64, off_i8).result

                        h_inc = arith.DivSIOp(off_i64, kw_const).result
                        w_inc = arith.RemSIOp(off_i64, kw_const).result

                        oh_i64 = arith.IndexCastOp(i64, oh).result
                        ow_i64 = arith.IndexCastOp(i64, ow).result

                        hbase = arith.SubIOp(
                            arith.MulIOp(oh_i64, sh_const).result, ph_const
                        ).result
                        wbase = arith.SubIOp(
                            arith.MulIOp(ow_i64, sw_const).result, pw_const
                        ).result

                        ih = arith.AddIOp(
                            hbase, arith.MulIOp(h_inc, dh_const).result
                        ).result
                        iw = arith.AddIOp(
                            wbase, arith.MulIOp(w_inc, dw_const).result
                        ).result

                        flat_idx = arith.AddIOp(
                            arith.MulIOp(ih, in_w_const).result, iw
                        ).result
                        memref.StoreOp(flat_idx, indices_memref.result, [n, oh, ow, c])
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])
        else:
            c_for = scf.ForOp(lb, c_ub, step)
            with ir.InsertionPoint(c_for.body):
                c = c_for.induction_variable
                oh_for = scf.ForOp(lb, oh_ub, step)
                with ir.InsertionPoint(oh_for.body):
                    oh = oh_for.induction_variable
                    ow_for = scf.ForOp(lb, ow_ub, step)
                    with ir.InsertionPoint(ow_for.body):
                        ow = ow_for.induction_variable

                        off_i8 = tensor.ExtractOp(offsets_tensor, [n, c, oh, ow]).result
                        off_i64 = arith.ExtSIOp(i64, off_i8).result

                        h_inc = arith.DivSIOp(off_i64, kw_const).result
                        w_inc = arith.RemSIOp(off_i64, kw_const).result

                        oh_i64 = arith.IndexCastOp(i64, oh).result
                        ow_i64 = arith.IndexCastOp(i64, ow).result

                        hbase = arith.SubIOp(
                            arith.MulIOp(oh_i64, sh_const).result, ph_const
                        ).result
                        wbase = arith.SubIOp(
                            arith.MulIOp(ow_i64, sw_const).result, pw_const
                        ).result

                        ih = arith.AddIOp(
                            hbase, arith.MulIOp(h_inc, dh_const).result
                        ).result
                        iw = arith.AddIOp(
                            wbase, arith.MulIOp(w_inc, dw_const).result
                        ).result

                        flat_idx = arith.AddIOp(
                            arith.MulIOp(ih, in_w_const).result, iw
                        ).result
                        memref.StoreOp(flat_idx, indices_memref.result, [n, c, oh, ow])
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    indices_type = ir.RankedTensorType.get(shape, i64)
    return bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    ).result


def low_memory_max_pool_with_offsets_op(
    node: LowMemoryMaxPoolWithOffsetsOp,
    symbol_table: dict[tuple[str, int], ir.Operation],
):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    if input_tensor is None:
        return

    input_shape = list(ir.RankedTensorType(input_tensor.type).shape)
    if any(dim < 0 for dim in input_shape):
        raise NotImplementedError(
            "_low_memory_max_pool_with_offsets requires static shapes"
        )

    dim = len(input_shape) - 2
    if dim not in (2, 3):
        raise NotImplementedError(
            "_low_memory_max_pool_with_offsets supports only 2d/3d"
        )

    kernel_size = _normalize_pool_param(node.args[1], dim, node.args[1])
    stride = _normalize_pool_param(
        node.args[2] if len(node.args) > 2 else None,
        dim,
        kernel_size,
    )
    padding = _normalize_pool_param(
        node.args[3] if len(node.args) > 3 else None,
        dim,
        [0] * dim,
    )
    dilation = _normalize_pool_param(
        node.args[4] if len(node.args) > 4 else None,
        dim,
        [1] * dim,
    )

    if dim == 2:
        pool_node = MaxPool2dWithIndicesOp()
        pool_node._arguments = list(node.args)
        values, indices = max_pool2d_with_indices_op(pool_node, symbol_table)
        offsets = _max_pool_indices_to_offsets_2d(
            indices,
            _pool2d_input_hw(input_shape),
            kernel_size,
            stride,
            padding,
            dilation,
        )
        return values, offsets

    pool_node = MaxPool3dOp()
    pool_node._arguments = list(node.args)
    values, indices = max_pool3d_op(pool_node, symbol_table)
    offsets = _max_pool_indices_to_offsets_3d(
        indices,
        input_shape[-3:],
        kernel_size,
        stride,
        padding,
        dilation,
    )
    return values, offsets


def max_pool2d_with_indices_op(
    node: MaxPool2dWithIndicesOp,
    symbol_table: dict[tuple[str, int], ir.Operation],
):
    """
    Import the max_pool2d_with_indices operation.
    From buddy MaxPool2dWithIndicesOp to MLIR operations using scf.for loops.
    aten.max_pool2d_with_indices(input, kernel_size, stride, padding,
                                  dilation, ceil_mode) -> (Tensor, Tensor)

    Returns both the max-pooled values and the indices of the max values.
    Uses scf.for loops since TOSA max_pool2d doesn't support returning indices.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    kernel_size = node.args[1]
    stride = node.args[2] if len(node.args) > 2 and node.args[2] else kernel_size
    padding = node.args[3] if len(node.args) > 3 else [0, 0]
    dilation = node.args[4] if len(node.args) > 4 else [1, 1]
    ceil_mode = node.args[5] if len(node.args) > 5 else False

    ssa_shape = [int(x) for x in ir.RankedTensorType(input_tensor.type).shape]
    src = node.args[0] if node.args else None
    meta = None
    if (
        src is not None
        and hasattr(src, "tensor_meta")
        and src.tensor_meta
        and "shape" in src.tensor_meta
    ):
        meta = [int(x) for x in src.tensor_meta["shape"]]
    if meta is not None:
        if len(meta) != 4:
            raise RuntimeError(
                f"max_pool2d_with_indices requires 4D input meta: {meta}"
            )
        N, C, H, W = meta
        nhwc = ssa_shape == [N, H, W, C]
        if not nhwc and ssa_shape != [N, C, H, W]:
            raise RuntimeError(
                f"max_pool2d_with_indices layout mismatch: ssa={ssa_shape} meta_nchw={meta}"
            )
    else:
        if len(ssa_shape) != 4:
            raise RuntimeError(
                f"max_pool2d_with_indices requires 4D input, got {ssa_shape}"
            )
        n, a, b, c = ssa_shape
        if a == b and c > a:
            N, H, W, C = n, a, b, c
            nhwc = True
        elif b == c and a > b:
            N, C, H, W = n, a, b, c
            nhwc = False
        else:
            raise RuntimeError(
                f"max_pool2d_with_indices cannot infer layout from ssa={ssa_shape}"
            )

    input_dtype = ir.RankedTensorType(input_tensor.type).element_type

    # Normalize kernel_size, stride, padding, dilation
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh = kernel_size[0]
        kw = kernel_size[1] if len(kernel_size) > 1 else kernel_size[0]

    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh = stride[0]
        sw = stride[1] if len(stride) > 1 else stride[0]

    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph = padding[0]
        pw = padding[1] if len(padding) > 1 else padding[0]

    if isinstance(dilation, int):
        dh, dw = dilation, dilation
    else:
        dh = dilation[0]
        dw = dilation[1] if len(dilation) > 1 else dilation[0]

    # Calculate output dimensions
    if ceil_mode:
        out_h = (H + 2 * ph - dh * (kh - 1) - 1 + sh - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1 + sw - 1) // sw + 1
    else:
        out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    output_shape = [N, out_h, out_w, C] if nhwc else [N, C, out_h, out_w]
    output_type = ir.RankedTensorType.get(output_shape, input_dtype)
    indices_dtype = ir.IntegerType.get_signless(64)
    indices_type = ir.RankedTensorType.get(output_shape, indices_dtype)

    # Create memrefs for output and indices
    output_memref_type = ir.MemRefType.get(output_shape, input_dtype)
    indices_memref_type = ir.MemRefType.get(output_shape, indices_dtype)

    output_memref = memref.AllocOp(output_memref_type, [], [])
    indices_memref = memref.AllocOp(indices_memref_type, [], [])

    # Initialize output with negative infinity
    neg_inf = arith.ConstantOp(
        input_dtype, ir.FloatAttr.get(input_dtype, float("-inf"))
    )
    linalg.fill(neg_inf.result, outs=[output_memref.result])

    # Initialize indices with zeros
    zero_idx = arith.ConstantOp(indices_dtype, ir.IntegerAttr.get(indices_dtype, 0))
    linalg.fill(zero_idx.result, outs=[indices_memref.result])

    # Convert input to memref
    input_memref = bufferization.ToBufferOp(
        ir.MemRefType.get(ssa_shape, input_dtype), input_tensor
    ).result

    # Create constants
    zero = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
    )
    one = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1)
    )
    n_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), N)
    )
    c_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), C)
    )
    out_h_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_h)
    )
    out_w_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), out_w)
    )
    kh_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kh)
    )
    kw_bound = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), kw)
    )

    # Constants for calculations
    sh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sh)
    )
    sw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), sw)
    )
    ph_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), ph)
    )
    pw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), pw)
    )
    dh_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dh)
    )
    dw_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), dw)
    )
    h_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), H)
    )
    w_const = arith.ConstantOp(
        ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), W)
    )
    w_i64 = arith.ConstantOp(indices_dtype, ir.IntegerAttr.get(indices_dtype, W))

    # Six nested loops: n, c, oh, ow, kh, kw
    n_loop = scf.ForOp(zero.result, n_bound.result, one.result)
    with ir.InsertionPoint(n_loop.body):
        n = n_loop.induction_variable

        c_loop = scf.ForOp(zero.result, c_bound.result, one.result)
        with ir.InsertionPoint(c_loop.body):
            c = c_loop.induction_variable

            oh_loop = scf.ForOp(zero.result, out_h_bound.result, one.result)
            with ir.InsertionPoint(oh_loop.body):
                oh = oh_loop.induction_variable

                ow_loop = scf.ForOp(zero.result, out_w_bound.result, one.result)
                with ir.InsertionPoint(ow_loop.body):
                    ow = ow_loop.induction_variable

                    # Base input position: oh * stride - padding
                    h_base = arith.MulIOp(oh, sh_const.result).result
                    h_base = arith.SubIOp(h_base, ph_const.result).result
                    w_base = arith.MulIOp(ow, sw_const.result).result
                    w_base = arith.SubIOp(w_base, pw_const.result).result

                    # Inner loops over kernel
                    kih_loop = scf.ForOp(zero.result, kh_bound.result, one.result)
                    with ir.InsertionPoint(kih_loop.body):
                        kih = kih_loop.induction_variable

                        kiw_loop = scf.ForOp(zero.result, kw_bound.result, one.result)
                        with ir.InsertionPoint(kiw_loop.body):
                            kiw = kiw_loop.induction_variable

                            # Calculate input position with dilation
                            ih = arith.AddIOp(
                                h_base,
                                arith.MulIOp(kih, dh_const.result).result,
                            ).result
                            iw = arith.AddIOp(
                                w_base,
                                arith.MulIOp(kiw, dw_const.result).result,
                            ).result

                            # Check bounds: 0 <= ih < H and 0 <= iw < W
                            ih_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, ih, zero.result
                            ).result
                            ih_lt_h = arith.CmpIOp(
                                arith.CmpIPredicate.slt, ih, h_const.result
                            ).result
                            iw_ge_0 = arith.CmpIOp(
                                arith.CmpIPredicate.sge, iw, zero.result
                            ).result
                            iw_lt_w = arith.CmpIOp(
                                arith.CmpIPredicate.slt, iw, w_const.result
                            ).result

                            h_valid = arith.AndIOp(ih_ge_0, ih_lt_h).result
                            w_valid = arith.AndIOp(iw_ge_0, iw_lt_w).result
                            in_bounds = arith.AndIOp(h_valid, w_valid).result

                            # If in bounds, load and compare
                            if_op = scf.IfOp(in_bounds, has_else=False)
                            with ir.InsertionPoint(if_op.then_block):
                                if nhwc:
                                    in_idx = [n, ih, iw, c]
                                    out_idx = [n, oh, ow, c]
                                else:
                                    in_idx = [n, c, ih, iw]
                                    out_idx = [n, c, oh, ow]
                                # Load input value
                                input_val = memref.LoadOp(input_memref, in_idx).result

                                # Load current max
                                current_max = memref.LoadOp(
                                    output_memref.result, out_idx
                                ).result

                                # Check if input > current_max
                                is_greater = arith.CmpFOp(
                                    arith.CmpFPredicate.OGT,
                                    input_val,
                                    current_max,
                                ).result

                                inner_if = scf.IfOp(is_greater, has_else=False)
                                with ir.InsertionPoint(inner_if.then_block):
                                    # Update max value
                                    memref.StoreOp(
                                        input_val,
                                        output_memref.result,
                                        out_idx,
                                    )

                                    # Calculate flattened index: ih * W + iw
                                    ih_i64 = arith.IndexCastOp(indices_dtype, ih).result
                                    iw_i64 = arith.IndexCastOp(indices_dtype, iw).result
                                    flat_idx = arith.AddIOp(
                                        arith.MulIOp(ih_i64, w_i64.result).result,
                                        iw_i64,
                                    ).result

                                    # Store index
                                    memref.StoreOp(
                                        flat_idx,
                                        indices_memref.result,
                                        out_idx,
                                    )

                                    scf.YieldOp([])

                                scf.YieldOp([])

                            scf.YieldOp([])

                        scf.YieldOp([])

                    scf.YieldOp([])

                scf.YieldOp([])

            scf.YieldOp([])

        scf.YieldOp([])

    # Convert memrefs back to tensors
    output_result = bufferization.ToTensorOp(
        output_type, output_memref.result, restrict=True
    )
    indices_result = bufferization.ToTensorOp(
        indices_type, indices_memref.result, restrict=True
    )

    return output_result, indices_result


def _quantize_input(activation, scale, nchw_to_nhwc=False):
    input_shape = [int(x) for x in ir.RankedTensorType(activation.type).shape]
    f32 = ir.F32Type.get()
    i8 = ir.IntegerType.get_signless(8)
    if ir.RankedTensorType(activation.type).element_type != f32:
        raise ValueError("Quantized input quantization requires FP32")
    scale = float(scale)
    if not numpy.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Quantized input quantization scale must be finite and positive"
        )
    if nchw_to_nhwc:
        if len(input_shape) != 4:
            raise ValueError("NCHW-to-NHWC quantization requires rank 4")
        n, c, h, w = input_shape
        output_shape = [n, h, w, c]
        dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
        input_map = ir.AffineMap.get(4, 0, [dims[0], dims[3], dims[1], dims[2]])
        output_map = ir.AffineMap.get_identity(4)
    else:
        output_shape = input_shape
        input_map = ir.AffineMap.get_identity(len(input_shape))
        output_map = input_map
    out_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    op = linalg.GenericOp(
        [out_type],
        [activation],
        [output],
        ir.ArrayAttr.get(
            [ir.AffineMapAttr.get(input_map), ir.AffineMapAttr.get(output_map)]
        ),
        ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(output_shape)
        ),
    )
    op.operation.attributes["buckyball.quant_f32_to_i8"] = ir.BoolAttr.get(True)
    op.operation.attributes["nchw_to_nhwc"] = ir.BoolAttr.get(nchw_to_nhwc)
    op.operation.attributes["scale"] = ir.FloatAttr.get(f32, 1.0 / scale)
    block = ir.Block.create_at_start(op.region, [f32, i8])
    block.append(linalg.YieldOp([block.arguments[1]]))
    return op.result


def quantized_conv2d_op(node, symbol_table):
    activation = symbol_table.get((str(node.args[0]), 0))
    weight = symbol_table.get((str(node.args[1]), 0))
    if activation is None or weight is None:
        raise ValueError(f"missing Quantized Conv2D operand for {node.name}")

    f32 = ir.F32Type.get()
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    activation_type = ir.RankedTensorType(activation.type)
    input_shape = [int(x) for x in activation_type.shape]
    n, cin, h, w = node._input_shape
    if activation_type.element_type == f32:
        if input_shape == [n, cin, h, w]:
            nchw_to_nhwc = True
        elif input_shape == [n, h, w, cin]:
            nchw_to_nhwc = False
        else:
            raise ValueError(
                f"FP32 Quantized Conv2D input layout mismatch for {node.name}: "
                f"expected {[n, cin, h, w]} or {[n, h, w, cin]}, got {input_shape}"
            )
        activation = _quantize_input(
            activation, node._input_scale, nchw_to_nhwc=nchw_to_nhwc
        )
    elif activation_type.element_type == i8:
        if input_shape != [n, h, w, cin]:
            raise ValueError(
                f"INT8 Quantized Conv2D input layout mismatch for {node.name}: "
                f"expected NHWC {[n, h, w, cin]}, got {input_shape}"
            )
    else:
        raise ValueError("Quantized Conv2D activation must be FP32 or INT8")

    depthwise = isinstance(node, QuantizedDepthwiseConv2dOp)
    weight_type = ir.RankedTensorType(weight.type)
    cout, weight_cin, kh, kw = node._weight_shape
    actual_weight_shape = [int(x) for x in weight_type.shape]
    if actual_weight_shape != list(node._weight_shape):
        raise ValueError(f"Quantized convolution weight shape mismatch for {node.name}")
    if (
        weight_type.element_type != i8
        or (depthwise and (weight_cin != 1 or cin != cout))
        or (not depthwise and cin != weight_cin)
    ):
        raise ValueError("Quantized Conv2D requires INT8 OIHW weight with matching Cin")

    out_n, out_c, out_h, out_w = node._output_shape
    if out_n != n or out_c != cout:
        raise ValueError(f"Quantized Conv2D output shape mismatch for {node.name}")
    scale_values = (
        node._dequant_scale
        if node._final_output and node._activation != 2
        else node._requant_scale
    )
    if len(node._bias_i32) != cout or len(scale_values) != cout:
        raise ValueError(f"Quantized Conv2D quant parameter mismatch for {node.name}")
    bias_type = ir.RankedTensorType.get([cout], i32)
    bias_attr = ir.DenseElementsAttr.get(
        numpy.asarray(node._bias_i32, dtype=numpy.int32), type=bias_type
    )
    bias = arith.ConstantOp(bias_type, bias_attr).result
    scale_type = ir.RankedTensorType.get([cout], f32)
    scale_attr = ir.DenseElementsAttr.get(
        numpy.asarray(scale_values, dtype=numpy.float32), type=scale_type
    )
    scale = arith.ConstantOp(scale_type, scale_attr).result
    lut_type = ir.RankedTensorType.get([len(node._lut_i8)], i8)
    lut_attr = ir.DenseElementsAttr.get(
        numpy.asarray(node._lut_i8, dtype=numpy.int8), type=lut_type
    )
    lut = arith.ConstantOp(lut_type, lut_attr).result
    output_shape = (
        [n, cout, out_h, out_w]
        if node._final_output and not node._output_nhwc
        else [n, out_h, out_w, cout]
    )
    output_element = f32 if node._final_output else i8
    out_type = ir.RankedTensorType.get(output_shape, output_element)
    output = tensor.EmptyOp(output_shape, output_element)
    dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
    zero = ir.AffineExpr.get_constant(0)
    output_map = ir.AffineMap.get_identity(4)
    maps = [
        ir.AffineMap.get(4, 0, [zero, zero, zero, zero]),
        ir.AffineMap.get(4, 0, [zero, zero, zero, zero]),
        ir.AffineMap.get(4, 0, [zero]),
        ir.AffineMap.get(4, 0, [zero]),
        ir.AffineMap.get(4, 0, [zero]),
        output_map,
    ]
    op = linalg.GenericOp(
        [out_type],
        [activation, weight, bias, scale, lut],
        [output],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(x) for x in maps]),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    i64 = ir.IntegerType.get_signless(64)
    marker = (
        "buckyball.quantized_conv2d_depthwise"
        if depthwise
        else "buckyball.quantized_conv2d"
    )
    op.operation.attributes[marker] = ir.BoolAttr.get(True)
    op.operation.attributes["stride"] = ir.IntegerAttr.get(i64, node._stride)
    op.operation.attributes["kernel"] = ir.IntegerAttr.get(i64, kh)
    op.operation.attributes["pad_low"] = ir.IntegerAttr.get(i64, node._padding)
    op.operation.attributes["pad_high"] = ir.IntegerAttr.get(i64, node._padding)
    op.operation.attributes["activation"] = ir.IntegerAttr.get(i64, node._activation)
    op.operation.attributes["output_scale"] = ir.FloatAttr.get(f32, node._output_scale)
    if getattr(node, "_lane_output_scales", []):
        op.operation.attributes["lane_output_scales"] = ir.DenseF32ArrayAttr.get(
            node._lane_output_scales
        )
    op.operation.attributes["final_output"] = ir.BoolAttr.get(node._final_output)
    op.operation.attributes["output_nhwc"] = ir.BoolAttr.get(node._output_nhwc)
    if node._output_nhwc or not node._final_output:
        op.operation.attributes["buddy.layout"] = ir.StringAttr.get("nhwc")
    block = ir.Block.create_at_start(op.region, [i8, i8, i32, f32, i8, output_element])
    block.append(linalg.YieldOp([block.arguments[5]]))
    return op.result


def quantized_matmul_op(node, symbol_table):
    activation = symbol_table.get((str(node.args[0]), 0))
    weight = symbol_table.get((str(node.args[1]), 0))
    if activation is None or weight is None:
        raise ValueError(f"missing Quantized MatMul operand for {node.name}")

    f32 = ir.F32Type.get()
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    activation_type = ir.RankedTensorType(activation.type)
    if activation_type.element_type == f32:
        activation = _quantize_input(activation, node._input_scale, nchw_to_nhwc=False)
        activation_type = ir.RankedTensorType(activation.type)
    weight_type = ir.RankedTensorType(weight.type)
    if (
        activation_type.element_type != i8
        or weight_type.element_type != i8
        or activation_type.rank != 2
        or weight_type.rank != 2
    ):
        raise ValueError("Quantized MatMul requires rank-2 INT8 activation and weight")
    m, k = [int(x) for x in activation_type.shape]
    n, wk = [int(x) for x in weight_type.shape]
    scale_values = (
        node._dequant_scale
        if node._final_output and node._activation != 2
        else node._requant_scale
    )
    if k != wk or len(node._bias_i32) != n or len(scale_values) != n:
        raise ValueError(f"Quantized MatMul shape mismatch for {node.name}")

    bias_type = ir.RankedTensorType.get([n], i32)
    bias_attr = ir.DenseElementsAttr.get(
        numpy.asarray(node._bias_i32, dtype=numpy.int32), type=bias_type
    )
    bias = arith.ConstantOp(bias_type, bias_attr).result
    scale_type = ir.RankedTensorType.get([n], f32)
    scale_attr = ir.DenseElementsAttr.get(
        numpy.asarray(scale_values, dtype=numpy.float32), type=scale_type
    )
    scale = arith.ConstantOp(scale_type, scale_attr).result
    lut_type = ir.RankedTensorType.get([len(node._lut_i8)], i8)
    lut_attr = ir.DenseElementsAttr.get(
        numpy.asarray(node._lut_i8, dtype=numpy.int8), type=lut_type
    )
    lut = arith.ConstantOp(lut_type, lut_attr).result
    output_element = f32 if node._final_output else i8
    out_type = ir.RankedTensorType.get([m, n], output_element)
    output = tensor.EmptyOp([m, n], output_element)
    dims = [ir.AffineExpr.get_dim(i) for i in range(3)]
    maps = [
        ir.AffineMap.get(3, 0, [dims[0], dims[2]]),
        ir.AffineMap.get(3, 0, [dims[1], dims[2]]),
        ir.AffineMap.get(3, 0, [dims[1]]),
        ir.AffineMap.get(3, 0, [dims[1]]),
        ir.AffineMap.get(3, 0, [ir.AffineExpr.get_constant(0)]),
        ir.AffineMap.get(3, 0, [dims[0], dims[1]]),
    ]
    op = linalg.GenericOp(
        [out_type],
        [activation, weight, bias, scale, lut],
        [output],
        ir.ArrayAttr.get([ir.AffineMapAttr.get(x) for x in maps]),
        ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                ir.Attribute.parse("#linalg.iterator_type<reduction>"),
            ]
        ),
    )
    op.operation.attributes["buckyball.quantized_matmul"] = ir.BoolAttr.get(True)
    op.operation.attributes["activation"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), node._activation
    )
    op.operation.attributes["output_scale"] = ir.FloatAttr.get(f32, node._output_scale)
    op.operation.attributes["final_output"] = ir.BoolAttr.get(node._final_output)
    block = ir.Block.create_at_start(op.region, [i8, i8, i32, f32, i8, output_element])
    block.append(linalg.YieldOp([block.arguments[5]]))
    return op.result


def quantized_max_pool2d_op(node, symbol_table):
    input_value = symbol_table.get((str(node.args[0]), 0))
    if input_value is None:
        raise ValueError(f"missing Quantized MaxPool2D input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    input_type = ir.RankedTensorType(input_value.type)
    if input_type.element_type != i8 or input_type.rank != 4:
        raise ValueError("Quantized MaxPool2D input must be rank-4 INT8 NHWC")
    n, h, w, c = [int(x) for x in input_type.shape]
    out_n, out_c, out_h, out_w = node._output_shape
    if (
        [n, c] != [out_n, out_c]
        or node._kernel <= 0
        or node._stride <= 0
        or node._padding < 0
        or (h + 2 * node._padding - node._kernel) // node._stride + 1 != out_h
        or (w + 2 * node._padding - node._kernel) // node._stride + 1 != out_w
    ):
        raise ValueError(f"Quantized MaxPool2D geometry mismatch for {node.name}")
    output_shape = (
        [out_n, out_c, out_h, out_w]
        if node._final_output
        else [out_n, out_h, out_w, out_c]
    )
    output_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    # This is a marker consumed by Tile lowering, like Quantized Conv. Iterate
    # over the output; input-sized loops overrun odd-sized pooling results.
    zero = ir.AffineExpr.get_constant(0)
    input_map = ir.AffineMap.get(4, 0, [zero, zero, zero, zero])
    output_map = ir.AffineMap.get_identity(4)
    op = linalg.GenericOp(
        [output_type],
        [input_value],
        [output],
        ir.ArrayAttr.get(
            [ir.AffineMapAttr.get(input_map), ir.AffineMapAttr.get(output_map)]
        ),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    i64 = ir.IntegerType.get_signless(64)
    op.operation.attributes["buckyball.quantized_max_pool2d"] = ir.BoolAttr.get(True)
    op.operation.attributes["kernel"] = ir.IntegerAttr.get(i64, node._kernel)
    op.operation.attributes["stride"] = ir.IntegerAttr.get(i64, node._stride)
    op.operation.attributes["padding"] = ir.IntegerAttr.get(i64, node._padding)
    op.operation.attributes["final_output"] = ir.BoolAttr.get(node._final_output)
    block = ir.Block.create_at_start(op.region, [i8, i8])
    block.append(linalg.YieldOp([block.arguments[1]]))
    return op.result


def quantized_global_avg_pool_op(node, symbol_table):
    input_value = symbol_table.get((str(node.args[0]), 0))
    if input_value is None:
        raise ValueError(f"missing Quantized global-average input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    input_type = ir.RankedTensorType(input_value.type)
    if input_type.element_type != i8 or input_type.rank != 4:
        raise ValueError("Quantized global-average input must be rank-4 INT8 NHWC")
    n, h, w, c = [int(x) for x in input_type.shape]
    if min(n, h, w, c) <= 0:
        raise ValueError("Quantized global-average shape must be positive")
    output_type = ir.RankedTensorType.get([n, 1, 1, c], i8)
    output = tensor.EmptyOp([n, 1, 1, c], i8)
    dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
    zero = ir.AffineExpr.get_constant(0)
    op = linalg.GenericOp(
        [output_type],
        [input_value],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)),
                ir.AffineMapAttr.get(
                    ir.AffineMap.get(4, 0, [dims[0], zero, zero, dims[3]])
                ),
            ]
        ),
        ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                ir.Attribute.parse("#linalg.iterator_type<reduction>"),
                ir.Attribute.parse("#linalg.iterator_type<reduction>"),
                ir.Attribute.parse("#linalg.iterator_type<parallel>"),
            ]
        ),
    )
    op.operation.attributes["buckyball.quantized_global_avg_pool"] = ir.BoolAttr.get(
        True
    )
    f32 = ir.F32Type.get()
    op.operation.attributes["input_scale"] = ir.FloatAttr.get(f32, node._input_scale)
    op.operation.attributes["output_scale"] = ir.FloatAttr.get(f32, node._output_scale)
    block = ir.Block.create_at_start(op.region, [i8, i8])
    block.append(linalg.YieldOp([block.arguments[1]]))
    return op.result


def quantized_elementwise_op(node, symbol_table, kind):
    lhs = symbol_table.get((str(node.args[0]), 0))
    rhs = symbol_table.get((str(node.args[1]), 0))
    if lhs is None or rhs is None:
        raise ValueError(f"missing Quantized INT8 {kind} input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    lhs_type = ir.RankedTensorType(lhs.type)
    rhs_type = ir.RankedTensorType(rhs.type)
    if lhs_type.element_type != i8 or rhs_type.element_type != i8:
        raise ValueError(
            f"Quantized INT8 {kind} operands must be INT8 for {node.name}: "
            f"{node.args[0]}={lhs_type}, {node.args[1]}={rhs_type}"
        )
    lhs_shape = [int(x) for x in lhs_type.shape]
    rhs_shape = [int(x) for x in rhs_type.shape]
    if len(lhs_shape) != 4 or len(rhs_shape) != 4:
        raise ValueError(f"Quantized INT8 {kind} operands must be rank 4")
    output_shape = [max(a, b) for a, b in zip(lhs_shape, rhs_shape)]
    if any(
        a not in (1, out) or b not in (1, out)
        for a, b, out in zip(lhs_shape, rhs_shape, output_shape)
    ):
        raise ValueError(f"Quantized INT8 {kind} operands are not broadcast-compatible")
    if kind == "add" and lhs_shape != rhs_shape:
        raise ValueError("Quantized residual add requires equal shapes")
    output_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
    zero = ir.AffineExpr.get_constant(0)
    lhs_map = ir.AffineMap.get(
        4,
        0,
        [zero if size == 1 else dims[i] for i, size in enumerate(lhs_shape)],
    )
    rhs_map = ir.AffineMap.get(
        4,
        0,
        [zero if size == 1 else dims[i] for i, size in enumerate(rhs_shape)],
    )
    op = linalg.GenericOp(
        [output_type],
        [lhs, rhs],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(lhs_map),
                ir.AffineMapAttr.get(rhs_map),
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)),
            ]
        ),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    op.operation.attributes[f"buckyball.quantized_int8_{kind}"] = ir.BoolAttr.get(True)
    f32 = ir.F32Type.get()
    op.operation.attributes["lhs_scale"] = ir.FloatAttr.get(f32, node._lhs_scale)
    op.operation.attributes["rhs_scale"] = ir.FloatAttr.get(f32, node._rhs_scale)
    op.operation.attributes["output_scale"] = ir.FloatAttr.get(f32, node._output_scale)
    op.operation.attributes["activation"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), node._activation
    )
    block = ir.Block.create_at_start(op.region, [i8, i8, i8])
    block.append(linalg.YieldOp([block.arguments[2]]))
    return op.result


def quantized_channel_slice_op(node, symbol_table):
    source = symbol_table.get((str(node.args[0]), 0))
    if source is None:
        raise ValueError(f"missing Quantized channel-slice input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    source_type = ir.RankedTensorType(source.type)
    if source_type.element_type != i8 or len(source_type.shape) != 4:
        raise ValueError(f"Quantized channel-slice requires rank-4 INT8 at {node.name}")
    n, c, h, w = node._output_shape
    output_shape = [n, h, w, c]
    output_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    source_slice = tensor.ExtractSliceOp(
        output_type,
        source,
        [],
        [],
        [],
        ir._denseI64ArrayAttr([0, 0, 0, node._offset], None),
        ir._denseI64ArrayAttr(output_shape, None),
        ir._denseI64ArrayAttr([1, 1, 1, 1], None),
    )
    op = linalg.GenericOp(
        [output_type],
        [source_slice],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)),
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)),
            ]
        ),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    op.operation.attributes["buckyball.quantized_channel_slice"] = ir.BoolAttr.get(True)
    op.operation.attributes["offset"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), 0
    )
    block = ir.Block.create_at_start(op.region, [i8, i8])
    block.append(linalg.YieldOp([block.arguments[0]]))
    return op.result


def quantized_channel_concat_op(node, symbol_table):
    inputs = [symbol_table.get((str(name), 0)) for name in node.args]
    if not inputs or any(value is None for value in inputs):
        raise ValueError(f"missing Quantized channel-concat input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    input_shapes = [list(ir.RankedTensorType(value.type).shape) for value in inputs]
    if any(
        ir.RankedTensorType(value.type).element_type != i8 or len(shape) != 4
        for value, shape in zip(inputs, input_shapes)
    ):
        raise ValueError(
            f"Quantized channel-concat requires rank-4 INT8 at {node.name}"
        )
    n, c, h, w = node._output_shape
    output_shape = [n, h, w, c]
    if (
        any(shape[:3] != [n, h, w] for shape in input_shapes)
        or sum(shape[3] for shape in input_shapes) != c
    ):
        raise ValueError(f"Quantized channel-concat shape mismatch at {node.name}")
    output_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
    zero = ir.AffineExpr.get_constant(0)
    maps = [
        ir.AffineMapAttr.get(ir.AffineMap.get(4, 0, [dims[0], dims[1], dims[2], zero]))
        for _ in inputs
    ]
    maps.append(ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)))
    op = linalg.GenericOp(
        [output_type],
        inputs,
        [output],
        ir.ArrayAttr.get(maps),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    op.operation.attributes["buckyball.quantized_channel_concat"] = ir.BoolAttr.get(
        True
    )
    op.operation.attributes["segments"] = ir.DenseI64ArrayAttr.get(
        [shape[3] for shape in input_shapes]
    )
    block = ir.Block.create_at_start(op.region, [i8] * (len(inputs) + 1))
    block.append(linalg.YieldOp([block.arguments[-1]]))
    return op.result


def quantized_resize_nearest_op(node, symbol_table):
    source = symbol_table.get((str(node.args[0]), 0))
    if source is None:
        raise ValueError(f"missing Quantized resize input for {node.name}")
    i8 = ir.IntegerType.get_signless(8)
    source_type = ir.RankedTensorType(source.type)
    if source_type.element_type != i8 or len(source_type.shape) != 4:
        raise ValueError(f"Quantized resize requires rank-4 INT8 at {node.name}")
    n, c, h, w = node._output_shape
    output_shape = [n, h, w, c]
    output_type = ir.RankedTensorType.get(output_shape, i8)
    output = tensor.EmptyOp(output_shape, i8)
    dims = [ir.AffineExpr.get_dim(i) for i in range(4)]
    zero = ir.AffineExpr.get_constant(0)
    op = linalg.GenericOp(
        [output_type],
        [source],
        [output],
        ir.ArrayAttr.get(
            [
                ir.AffineMapAttr.get(
                    ir.AffineMap.get(4, 0, [dims[0], zero, zero, dims[3]])
                ),
                ir.AffineMapAttr.get(ir.AffineMap.get_identity(4)),
            ]
        ),
        ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * 4),
    )
    op.operation.attributes["buckyball.quantized_resize_nearest"] = ir.BoolAttr.get(
        True
    )
    op.operation.attributes["scale_h"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), node._scale_h
    )
    op.operation.attributes["scale_w"] = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(64), node._scale_w
    )
    block = ir.Block.create_at_start(op.region, [i8, i8])
    block.append(linalg.YieldOp([block.arguments[0]]))
    return op.result


def quantized(lower):
    def emit(node, symbols):
        result = lower(node, symbols)
        if node.quantization_trace_meta is not None:
            from types import SimpleNamespace

            result.owner.attributes["trace_node"] = ir.DictAttr.get(
                _trace_attrs(SimpleNamespace(trace_meta=node.quantization_trace_meta))
            )
        return result

    return emit


from buddy.compiler.ops.linalg import ops_registry as upstream_registry

ops_registry = {
    **upstream_registry,
    "QuantizedMulOp": quantized(
        lambda node, symbols: quantized_elementwise_op(node, symbols, "mul")
    ),
    "QuantizedAddOp": quantized(
        lambda node, symbols: quantized_elementwise_op(node, symbols, "add")
    ),
    "QuantizedConv2dOp": quantized(quantized_conv2d_op),
    "QuantizedDepthwiseConv2dOp": quantized(quantized_conv2d_op),
    "QuantizedMatmulOp": quantized(quantized_matmul_op),
    "QuantizedMaxPool2dOp": quantized(quantized_max_pool2d_op),
    "QuantizedGlobalAvgPoolOp": quantized(quantized_global_avg_pool_op),
    "QuantizedChannelSliceOp": quantized(quantized_channel_slice_op),
    "QuantizedChannelConcatOp": quantized(quantized_channel_concat_op),
    "QuantizedResizeNearestOp": quantized(quantized_resize_nearest_op),
    "CatOp": cat_op,
    "SplitOp": split_op,
    "MaxPool2dWithIndicesOp": max_pool2d_with_indices_op,
    "LowMemoryMaxPoolWithOffsetsOp": low_memory_max_pool_with_offsets_op,
}
