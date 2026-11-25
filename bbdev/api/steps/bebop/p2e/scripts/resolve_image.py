import glob
import os

from utils.path import design_output_root


def resolve_image(bbdir: str, image_name: str, design: str, selection: str) -> str:
    """The hex of a workload built for the simulation entry: a baremetal
    program's <name>.hex (workload --tohex), a Linux image's <name>-ddr.hex
    (the DDR image of the kernel and its model)."""
    image_name = image_name.replace(r"\_", "_")
    if os.path.isabs(image_name):
        raise ValueError(
            f"image must be a logical name, not an absolute path: {image_name!r}"
        )
    workloads = os.path.join(
        design_output_root(bbdir, design, selection=selection), "workloads"
    )
    matches = [
        match
        for name in (f"{image_name}.hex", f"{image_name}-ddr.hex")
        for match in glob.glob(os.path.join(workloads, "**", name), recursive=True)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one image for {image_name!r} under {workloads}, "
            f"found {len(matches)}"
        )
    return matches[0]
