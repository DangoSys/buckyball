import glob
import os


def resolve_image(bbdir: str, image_name: str, design: str) -> str:
    image_name = image_name.replace(r"\_", "_")
    if os.path.isabs(image_name):
        raise ValueError(
            f"image must be a logical name, not an absolute path: {image_name!r}"
        )
    matches = glob.glob(
        os.path.join(
            bbdir,
            "bb-tests",
            "output",
            design,
            "workloads",
            "**",
            f"{image_name}.hex",
        ),
        recursive=True,
    )
    kernel_image = os.path.join(
        bbdir, "stack", "output", "kernel", design, f"{image_name}.hex"
    )
    if os.path.isfile(kernel_image):
        matches.append(kernel_image)
    if len(matches) != 1:
        raise ValueError(
            f"expected one image for {image_name!r} under "
            f"bb-tests/output/{design}/workloads/ or stack/output/kernel/{design}/, "
            f"found {len(matches)}"
        )
    return matches[0]
