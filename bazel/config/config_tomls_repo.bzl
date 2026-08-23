"""Expose every chip/core/ball config TOML as a filegroup for toml2json inputs."""

def _bb_config_tomls_repo_impl(repo_ctx):
    root = repo_ctx.path(Label("//:MODULE.bazel")).dirname
    examples = root.get_child("examples")
    if not examples.exists:
        fail("examples/ does not exist")
    repo_ctx.watch(examples)

    srcs = []
    for top_name in ("chips", "cores", "balls"):
        top = examples.get_child(top_name)
        if not top.exists:
            continue
        repo_ctx.watch(top)
        _collect_tomls(repo_ctx, root, top, srcs)

    if not srcs:
        fail("no config TOML files found under examples/{chips,cores,balls}")

    lines = [
        'package(default_visibility = ["//visibility:public"])',
        "",
        "filegroup(",
        '    name = "tomls",',
        "    srcs = [",
    ]
    for s in sorted(srcs):
        lines.append('        "%s",' % s)
    lines += [
        "    ],",
        ")",
        "",
    ]
    repo_ctx.file("BUILD.bazel", "\n".join(lines))
    repo_ctx.file("WORKSPACE", "")

def _collect_tomls(repo_ctx, root, start, srcs):
    stack = [start]
    for _ in range(100000):
        if not stack:
            return
        cur = stack.pop()
        if not cur.exists or not cur.is_dir:
            continue
        for entry in cur.readdir():
            name = entry.basename
            if name.startswith("."):
                continue
            if name in ("target", "build", "verify", "node_modules"):
                continue
            if name == "Cargo.toml" or name == "Cargo.lock":
                continue
            if name.endswith(".toml"):
                if not entry.is_dir:
                    rel = _rel_from_root(root, entry)
                    dest = "files/" + rel.replace("/", "__")
                    repo_ctx.symlink(entry, dest)
                    srcs.append(dest)
                continue
            if entry.is_dir:
                stack.append(entry)
    fail("config toml walk exceeded bound under %s" % start)

def _rel_from_root(root, path):
    root_s = str(root)
    path_s = str(path)
    prefix = root_s + "/"
    if not path_s.startswith(prefix):
        fail("path escapes repo root: %s" % path_s)
    return path_s[len(prefix):]

bb_config_tomls_repo = repository_rule(
    implementation = _bb_config_tomls_repo_impl,
    local = True,
)
