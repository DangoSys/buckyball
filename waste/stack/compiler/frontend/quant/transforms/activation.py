from __future__ import annotations

import numpy as np
import torch


def _activation_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("activation tensor is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("activation tensor contains NaN or infinity")
    return values


def activation_max_abs(values: np.ndarray) -> np.float32:
    values = _activation_array(values)
    return np.float32(np.max(np.abs(values)))


def activation_scale(values: np.ndarray) -> np.float32:
    max_abs = activation_max_abs(values)
    if max_abs == 0.0:
        return np.float32(1.0)
    return np.float32(max_abs / np.float32(127.0))


def calibrate_layers(model: torch.nn.Module, batches) -> dict:
    """Aggregate layer ranges across all batches before converting to scales."""
    ranges, calls = {}, {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue

        def record(layer, inputs, output, weight_name=f"{name}.weight"):
            if (
                len(inputs) != 1
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                raise ValueError(f"unsupported calibration signature for {weight_name}")
            values = _activation_array(output.detach().cpu().numpy())
            axis = values.ndim - 1 if isinstance(layer, torch.nn.Linear) else 1
            axes = tuple(i for i in range(values.ndim) if i != axis)
            hardswish = (
                values * np.clip(values + np.float32(3.0), 0.0, 6.0) / np.float32(6.0)
            )
            observed = (
                activation_max_abs(inputs[0].detach().cpu().numpy()),
                activation_max_abs(values),
                activation_max_abs(np.maximum(values, np.float32(0.0))),
                np.max(np.abs(values), axis=axes),
                np.max(np.abs(hardswish), axis=axes),
            )
            index = calls.get(weight_name, 0)
            calls[weight_name] = index + 1
            records = ranges.setdefault(weight_name, [])
            if index == len(records):
                records.append(observed)
            else:
                records[index] = tuple(
                    np.maximum(a, b)
                    for a, b in zip(records[index], observed, strict=True)
                )

        handles.append(module.register_forward_hook(record))
    expected = None
    try:
        with torch.no_grad():
            for inputs in batches:
                calls.clear()
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)
                if expected is None:
                    expected = dict(calls)
                elif calls != expected:
                    raise ValueError(
                        "Calibration batches executed different model layers"
                    )
    finally:
        for handle in handles:
            handle.remove()
    if not ranges:
        raise ValueError("calibration found no Conv2d or Linear activations")
    return {
        name: [
            tuple(
                np.where(value == 0, np.float32(1.0), value / np.float32(127.0)).astype(
                    np.float32
                )
                for value in record
            )
            for record in records
        ]
        for name, records in ranges.items()
    }


def quantize_activation_with_scale(
    values: np.ndarray, da: float | np.float32
) -> np.ndarray:
    values = _activation_array(values)
    da = np.float32(da)
    if not np.isfinite(da) or da <= 0.0:
        raise ValueError("Da must be finite and positive")
    return np.clip(np.rint(values / da), -128, 127).astype(np.int8)


def quantize_activation(values: np.ndarray) -> tuple[np.ndarray, np.float32]:
    da = activation_scale(values)
    return quantize_activation_with_scale(values, da), da


def quantize_activation_banks(
    values: np.ndarray, bank_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    values = _activation_array(values)
    if values.ndim != 2:
        raise ValueError("bank activation quantization requires a rank-2 matrix")
    if type(bank_rows) is not int or bank_rows <= 0:
        raise ValueError("bank_rows must be a positive integer")
    quantized = np.empty(values.shape, dtype=np.int8)
    bank_count = (values.shape[0] + bank_rows - 1) // bank_rows
    scales = np.empty(bank_count, dtype=np.float32)
    for bank, row in enumerate(range(0, values.shape[0], bank_rows)):
        bank_quantized, da = quantize_activation(values[row : row + bank_rows])
        quantized[row : row + bank_rows] = bank_quantized
        scales[bank] = da
    return quantized, scales


def dequantize_accumulator(
    accumulator: np.ndarray, da: float | np.float32, dw: float | np.ndarray
) -> np.ndarray:
    accumulator = np.asarray(accumulator, dtype=np.int32)
    da = np.float32(da)
    dw = np.asarray(dw, dtype=np.float32)
    if not np.isfinite(da) or da <= 0.0:
        raise ValueError("Da must be finite and positive")
    if not np.all(np.isfinite(dw)) or np.any(dw <= 0.0):
        raise ValueError("Dw must be finite and positive")
    if dw.ndim > 1 or (dw.ndim == 1 and accumulator.ndim == 0):
        raise ValueError("Dw must be scalar or a last-dimension scale vector")
    if dw.ndim == 1 and accumulator.shape[-1] != dw.shape[0]:
        raise ValueError("Dw channel count does not match accumulator")
    return accumulator.astype(np.float32) * da * dw


def w8a8_matmul(
    activations: np.ndarray,
    weights_i8: np.ndarray,
    dw: float | np.ndarray,
    bank_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    activations = _activation_array(activations)
    weights_i8 = np.asarray(weights_i8)
    if activations.ndim != 2 or weights_i8.ndim != 2:
        raise ValueError("W8A8 matmul requires rank-2 activation and weight tensors")
    if weights_i8.dtype != np.int8:
        raise ValueError("W8A8 weights must be INT8")
    if activations.shape[1] != weights_i8.shape[0]:
        raise ValueError("W8A8 matmul K dimensions do not match")
    if type(bank_rows) is not int or bank_rows <= 0:
        raise ValueError("bank_rows must be a positive integer")
    quantized, da_values = quantize_activation_banks(activations, bank_rows)
    result = np.empty((activations.shape[0], weights_i8.shape[1]), dtype=np.float32)
    for bank, row in enumerate(range(0, activations.shape[0], bank_rows)):
        accumulator = quantized[row : row + bank_rows].astype(
            np.int32
        ) @ weights_i8.astype(np.int32)
        result[row : row + bank_rows] = dequantize_accumulator(
            accumulator, da_values[bank], dw
        )
    return result, da_values
