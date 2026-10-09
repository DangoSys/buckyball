"""Prepare pinned sources without changing the enclosing Buckyball checkout.

Requires Git and curl. Existing exported files are verified, never overwritten.
Use --repository to reuse a local Git repository containing the baseline commit.
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.check_output(args, timeout=600)


def save(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite changed source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def prepare_upstream(repository):
    manifest = json.loads((ROOT / "UPSTREAM_MANIFEST.json").read_text())
    commit = manifest["commit"]
    if repository is None:
        try:
            repository = Path(
                run("git", "-C", str(ROOT), "rev-parse", "--show-toplevel")
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            repository = ROOT / "build" / "baseline.git"
            if not repository.exists():
                repository.mkdir(parents=True)
                run("git", "init", "--bare", str(repository))
    if subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", commit], capture_output=True
    ).returncode:
        run(
            "git",
            "-C",
            str(repository),
            "fetch",
            "--no-tags",
            "--depth=1",
            manifest["upstream_repository"],
            commit,
        )
    requests = "".join(f"{commit}:{name}\n" for name in manifest["files"]).encode()
    output = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "--batch"],
        input=requests,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    stream = io.BytesIO(output)
    for name, expected in manifest["files"].items():
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(f"Missing Git blob: {name}")
        data = stream.read(int(header[2]))
        assert stream.read(1) == b"\n"
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Original source hash mismatch: {name}")
        save(ROOT / "upstream" / name, data)
    print(f"Verified {len(manifest['files'])} original Buckyball files at {commit}")


def prepare_dependencies():
    lock = json.loads((ROOT / "dependencies.json").read_text())
    cache = ROOT / "build" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for name, spec in lock.items():
        archive = cache / f"{name}-{spec['commit']}.tar.gz"
        if not archive.exists():
            temporary = archive.with_suffix(".partial")
            url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['commit']}"
            run(
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
                "--max-time",
                "300",
                "--output",
                str(temporary),
                url,
            )
            temporary.replace(archive)
        destination = ROOT / "thirdparty" / name
        count = 0
        with tarfile.open(archive) as source:
            for member in source:
                if not member.isfile():
                    continue
                relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                ):
                    raise ValueError(f"Unsafe archive path: {member.name}")
                subtree = spec.get("subtree")
                if subtree and not (
                    str(relative).startswith(subtree + "/")
                    or relative.name.upper().startswith(
                        ("LICENSE", "COPYING", "NOTICE")
                    )
                ):
                    continue
                target = (destination / str(relative)).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Archive path escapes destination: {member.name}")
                save(target, source.extractfile(member).read())
                count += 1
        print(f"Verified/exported {name}: {count} files at {spec['commit']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    arguments = parser.parse_args()
    prepare_upstream(arguments.repository)
    prepare_dependencies()
