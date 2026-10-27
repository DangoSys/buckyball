import glob
import os

from utils.path import design_output_root


def resolve_image(bbdir: str, image_name: str, design: str, selection: str) -> str:
    """The hex image of a workload or kernel built for the simulation entry."""
    image_name = image_name.replace(r"\_", "_")
    if os.path.isabs(image_name):
        raise ValueError(
            f"image must be a logical name, not an absolute path: {image_name!r}"
        )
    workloads = os.path.join(
        design_output_root(bbdir, design, selection=selection), "workloads"
    )
    matches = glob.glob(
        os.path.join(workloads, "**", f"{image_name}.hex"), recursive=True
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one image for {image_name!r} under {workloads}, "
            f"found {len(matches)}"
        )
    return matches[0]
