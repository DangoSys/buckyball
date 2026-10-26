"""Calibrate symmetric INT8 scales from model inputs."""

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Subset

from stack.compiler.frontend.quant.offline.weights import dequantize_weights


def calibrate_activations(
    model,
    quantized,
    dataset,
    *,
    samples,
    layers,
    batch_size=64,
    granularities=None,
):
    """Calibrate selected module inputs on a copy using dequantized weights.

    The model supplies its dataset and graph-node-to-module mapping. Sampling
    and batching are fixed here so repeated calibration is deterministic.
    """
    model, loader = _calibration_batches(model, quantized, dataset, samples, batch_size)
    modules = dict(model.named_modules())
    maxima = dict.fromkeys(layers)
    observed = set()
    handles = []
    selected = [(name, modules[layer]) for name, layer in layers.items()]
    for name, module in selected:

        def record(module, inputs, *, name=name):
            values = inputs[0].detach()
            if values.dtype != torch.float32 or not torch.isfinite(values).all():
                raise ValueError(f"{name}: expected finite FP32 activations")
            if granularities is not None and granularities[name] == "channel":
                maximum = values.abs().amax(dim=(0, 2, 3)).cpu()
            else:
                maximum = values.abs().amax().cpu()
            maxima[name] = (
                maximum
                if maxima[name] is None
                else torch.maximum(maxima[name], maximum)
            )
            observed.add(name)

        handles.append(module.register_forward_pre_hook(record))
    try:
        with torch.no_grad():
            for inputs, _ in loader:
                if isinstance(inputs, Mapping):
                    model(**inputs)
                else:
                    model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if observed != set(layers) or not layers:
        raise ValueError("Calibration did not observe every selected layer")
    result = {}
    for name, maximum in maxima.items():
        scale = torch.where(maximum == 0, 1.0, maximum / 127.0)
        result[name] = float(scale) if scale.ndim == 0 else scale.tolist()
    return result


def _calibration_batches(model, quantized, dataset, samples, batch_size=64):
    model = deepcopy(model).eval()
    if quantized is not None:
        restored = dequantize_weights(quantized)
        model.load_state_dict({name: restored[name] for name in model.state_dict()})
    if not 0 < samples <= len(dataset):
        raise ValueError("Calibration sample count must be within the dataset size")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[
        :samples
    ]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size)
    return model, loader


def calibrate_layer_ranges(model, dataset, *, samples, batch_size=64, quantized=None):
    """Calibrate integer backend input/output ranges on deterministic dataset samples."""
    from stack.compiler.frontend.quant.transforms.activation import calibrate_layers

    model, loader = _calibration_batches(model, quantized, dataset, samples, batch_size)
    return calibrate_layers(model, (inputs for inputs, _ in loader))
