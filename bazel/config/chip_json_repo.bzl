_CHIP_JSON = "//bazel/configparse:chip_json.py"

def _child(root, rel):
    path = root
    for part in rel.split("/"):
        if part:
            path = path.get_child(part)
    return path

def _watch_includes(repo_ctx, root, data):
    includes = data.get("includes")
    if type(includes) != "list":
        fail("chip_json output missing includes")
    for rel in includes:
        repo_ctx.watch(_child(root, rel))

def ensure_chip_bundle_pb2(repo_ctx, root):
    """Generate chip_bundle_pb2.py for repository rules that run chip_json.py."""
    out = repo_ctx.path("chip_bundle_pb2.py")
    if out.exists:
        return str(repo_ctx.path("."))

    proto_dir = root.get_child("bazel").get_child("proto")
    proto = proto_dir.get_child("chip_bundle.proto")
    repo_ctx.watch(proto)
    protoc = repo_ctx.which("protoc")
    if not protoc:
        fail("protoc not on PATH. Enter nix develop.")
    result = repo_ctx.execute([
        protoc,
        "-I" + str(proto_dir),
        "--python_out=" + str(repo_ctx.path(".")),
        str(proto),
    ])
    if result.return_code != 0:
        fail("protoc chip_bundle.proto failed (exit %s):\n%s%s" % (
            result.return_code,
            result.stdout,
            result.stderr,
        ))
    if not out.exists:
        fail("protoc did not write chip_bundle_pb2.py")
    return str(repo_ctx.path("."))

def run_chip_json(repo_ctx, root, out_name, extra_argv, what, pb2_dir):
    script = repo_ctx.path(Label(_CHIP_JSON))
    repo_ctx.watch(script)
    python = repo_ctx.which("python3")
    if not python:
        fail("python3 not on PATH. Enter nix develop.")
    out = repo_ctx.path(out_name)
    configparse = str(script.dirname)
    result = repo_ctx.execute(
        [
            str(python),
            str(script),
            "--repo",
            str(root),
        ] + extra_argv + [
            "--out",
            str(out),
        ],
        environment = {
            "PYTHONPATH": "%s:%s" % (pb2_dir, configparse),
        },
    )
    if result.return_code != 0:
        fail("chip_json %s failed (exit %s):\n%s%s" % (
            what,
            result.return_code,
            result.stdout,
            result.stderr,
        ))
    data = json.decode(repo_ctx.read(out))
    _watch_includes(repo_ctx, root, data)
    return data
