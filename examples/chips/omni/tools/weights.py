import argparse
import dataclasses
import json
import tomllib
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def plan(metadata, config):
    chips = config["chips"]
    stages = [chip for chip in chips if chip["role"] == "thinker"]
    media = [chip for chip in chips if chip["role"] == "media"]
    tiles = config["thinker"]["tensor_tiles"]
    if not stages or len(media) != 1 or tiles != [0, 1, 2, 3]:
        raise ValueError("Qwen3 Omni placement requires TP4 tiles and one media chip")
    names = [
        name for info in metadata["files_metadata"].values() for name in info["tensors"]
    ]
    layers = (
        max(
            int(name.split(".")[3])
            for name in names
            if name.startswith("thinker.model.layers.")
        )
        + 1
    )
    covered = [layer for chip in stages for layer in range(*chip["layers"])]
    if covered != list(range(layers)):
        raise ValueError("Thinker chips must cover every layer exactly once in order")
    totals = [
        {
            "chip": c["id"],
            "capacity_bytes": c["memory_mib"] << 20,
            "checkpoint_bytes": 0,
            "mxfp8_k128_estimate_bytes": 0,
            "mxfp8_k512_estimate_bytes": 0,
        }
        for c in chips
    ]
    components = defaultdict(int)
    tensors = []
    for filename, info in sorted(metadata["files_metadata"].items()):
        for name, tensor in sorted(info["tensors"].items()):
            shape = tensor["shape"]
            original_bytes = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            component = ".".join(name.split(".")[:2])
            components[component] += original_bytes
            entry = {
                "name": name,
                "file": filename,
                "dtype": tensor["dtype"],
                "shape": shape,
                "source_bytes": original_bytes,
                "placements": [],
            }
            if name.startswith(("thinker.model.", "thinker.lm_head.")):
                if name.startswith("thinker.model.layers."):
                    layer = int(name.split(".")[3])
                    chip = next(
                        chip
                        for chip in stages
                        if chip["layers"][0] <= layer < chip["layers"][1]
                    )
                elif name == "thinker.model.embed_tokens.weight":
                    chip = stages[0]
                elif name in ("thinker.model.norm.weight", "thinker.lm_head.weight"):
                    chip = stages[-1]
                else:
                    raise ValueError(f"No layer placement for {name}")
                selected = [(chip, tile, rank) for rank, tile in enumerate(tiles)]
                if name.endswith(
                    (
                        "q_proj.weight",
                        "k_proj.weight",
                        "v_proj.weight",
                        "gate_proj.weight",
                        "up_proj.weight",
                        "embed_tokens.weight",
                        "lm_head.weight",
                    )
                ):
                    axis = 0
                elif name.endswith(("o_proj.weight", "down_proj.weight")):
                    axis = 1
                elif len(shape) == 1 or name.endswith("mlp.gate.weight"):
                    axis = None
                else:
                    raise ValueError(f"No partition rule for {name}: {shape}")
            elif name.startswith(
                ("thinker.audio_tower.", "thinker.visual.", "talker.", "code2wav.")
            ):
                if name.startswith("thinker.audio_tower."):
                    tile = config["media"]["audio_tile"]
                elif name.startswith("thinker.visual."):
                    tile = config["media"]["vision_tile"]
                elif name.startswith("talker."):
                    tile = config["media"]["talker_tile"]
                else:
                    tile = config["media"]["code2wav_tile"]
                selected, axis = [(media[0], tile, 0)], None
            else:
                raise ValueError(f"Unknown Qwen3 Omni component: {name}")
            for chip, tile, rank in selected:
                shard_shape = list(shape)
                start = 0
                if axis is not None:
                    if shape[axis] % len(tiles):
                        raise ValueError(f"Cannot split {name} along axis {axis}")
                    shard_shape[axis] //= len(tiles)
                    start = rank * shard_shape[axis]
                size = original_bytes if axis is None else original_bytes // len(tiles)
                amounts = {
                    "checkpoint_bytes": size,
                    "mxfp8_k128_estimate_bytes": size,
                    "mxfp8_k512_estimate_bytes": size,
                }
                # Estimate only Thinker projection/expert matrices. Encoders, routing,
                # embeddings, Talker and waveform decode retain checkpoint precision.
                quantized = axis is not None and not name.endswith(
                    ("embed_tokens.weight", "lm_head.weight")
                )
                if quantized:
                    if len(shard_shape) != 2:
                        raise ValueError(f"Expected linear matrix: {name}")
                    rows, width = shard_shape
                    for tile_k in (128, 512):
                        elements = ((rows + 15) // 16 * 16) * (
                            (width + tile_k - 1) // tile_k * tile_k
                        )
                        amounts[f"mxfp8_k{tile_k}_estimate_bytes"] = (
                            elements + elements // 32
                        )
                totals[chip["id"]]["checkpoint_bytes"] += size
                for tile_k in (128, 512):
                    key = f"mxfp8_k{tile_k}_estimate_bytes"
                    totals[chip["id"]][key] += amounts[key]
                entry["placements"].append(
                    {
                        "chip": chip["id"],
                        "tile": tile,
                        "axis": axis,
                        "start": start,
                        "shape": shard_shape,
                        **amounts,
                    }
                )
            tensors.append(entry)
    for chip in totals:
        for kind in ("checkpoint", "mxfp8_k128_estimate", "mxfp8_k512_estimate"):
            chip[f"{kind}_remaining_bytes"] = (
                chip["capacity_bytes"] - chip[f"{kind}_bytes"]
            )
    return {
        "model": config["model"],
        "revision": config["revision"],
        "source_bytes": sum(components.values()),
        "components": dict(components),
        "chips": totals,
        "tensors": tensors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "configs/system.toml").open("rb") as source:
        config = tomllib.load(source)
    if args.metadata:
        data = json.loads(args.metadata.read_text())
        if data["repo"] != config["model"] or data["revision"] != config["revision"]:
            raise ValueError("Metadata does not match the pinned checkpoint")
        metadata = data["metadata"]
    else:
        metadata = dataclasses.asdict(
            HfApi().get_safetensors_metadata(
                config["model"], revision=config["revision"]
            )
        )
    result = plan(metadata, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Checkpoint tensors: {len(result['tensors'])}, bytes: {result['source_bytes']:,}"
    )
    for chip in result["chips"]:
        print(
            f"chip {chip['chip']}: checkpoint {chip['checkpoint_bytes'] / 2**30:.3f} GiB; "
            f"MXFP8 K128 estimate {chip['mxfp8_k128_estimate_bytes'] / 2**30:.3f} GiB; "
            f"K512 estimate {chip['mxfp8_k512_estimate_bytes'] / 2**30:.3f} GiB"
        )


if __name__ == "__main__":
    main()
