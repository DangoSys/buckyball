load(":chips_repo.bzl", "bb_chips_repo")
load(":config_tomls_repo.bzl", "bb_config_tomls_repo")
load(":cores_repo.bzl", "bb_cores_repo")

def _bb_chips_impl(module_ctx):
    bb_chips_repo(name = "bb_chips")

bb_chips = module_extension(implementation = _bb_chips_impl)

def _bb_cores_impl(module_ctx):
    bb_cores_repo(name = "bb_cores")

bb_cores = module_extension(implementation = _bb_cores_impl)

def _bb_config_tomls_impl(module_ctx):
    bb_config_tomls_repo(name = "bb_config_tomls")

bb_config_tomls = module_extension(implementation = _bb_config_tomls_impl)
