{ pkgs }:

{
  # Python and pip packages
  python3 = pkgs.python3;

  # Python packages
  python3Packages = pkgs.python3.withPackages (ps: with ps; [
    # bbdev
    python-dotenv
    httpx
    mcp
    redis
    httpx-sse
    requests

    # pre-commit hooks
    pre-commit-hooks

    # compiler
    torch
    numpy
    transformers
    tokenizers
    sentencepiece
    accelerate
    protobuf
    pybind11
    torchvision
    tabulate
    datasets
    soundfile
    librosa
    pyyaml
    certifi
    idna
    diffusers
    nanobind

    # testing (sardine)
    pytest
    pytest-html
    pytest-xdist
    pytest-cov
    allure-pytest
    colorlog
  ]);
}
