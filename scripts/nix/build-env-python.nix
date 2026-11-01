{ pkgs }:

let
  # firesim uses Fabric 1.x API (fabric.api), but nixpkgs only has Fabric 2.x/3.x.
  # firesim switched to fab-classic (a maintained Fabric 1.x fork) since 2023.
  # Build it from PyPI as a custom derivation.
  fab-classic = pkgs.python3Packages.buildPythonPackage rec {
    pname = "fab-classic";
    version = "1.19.2";
    format = "setuptools";

    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      hash = "sha256-kn5JLdNQhUoDzzzeIs6r/LDesY7Byc4RHprNm+5RFFg=";
    };

    propagatedBuildInputs = with pkgs.python3Packages; [
      paramiko
    ];

    # No tests in PyPI tarball
    doCheck = false;
  };

  # lerobot carries the VLA policies (SmolVLA, pi0.5) the model recipes export.
  # The test inputs of gymnasium pull tensorflow-bin, which has no build for
  # this Python; torchcodec only decodes dataset videos, which no recipe reads;
  # the cmake wheel it lists would shadow the toolchain's cmake.
  python3 = pkgs.python3.override {
    packageOverrides = self: super: {
      gymnasium = super.gymnasium.overridePythonAttrs (o: {
        doCheck = false;
        nativeCheckInputs = [ ];
      });
      lerobot = super.lerobot.overridePythonAttrs (o: {
        dependencies = builtins.filter (d: !builtins.elem d.pname [ "torchcodec" "cmake" ]) o.dependencies;
        doCheck = false;
        dontCheckRuntimeDeps = true;
        pythonImportsCheck = [ ];
      });
    };
  };
in
{
  # Python and pip packages
  inherit python3;

  # Python packages
  python3Packages = python3.withPackages (ps: with ps; [
    # bbdev
    pydantic
    python-dotenv
    httpx
    mcp
    redis
    httpx-sse
    requests
    pysocks
    socksio
    matplotlib

    # pre-commit hooks (language: system use)
    black
    flake8
    pre-commit-hooks

    # compiler
    tomli-w
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
    ultralytics
    lerobot
    av
    num2words

    # testing
    pytest
    pytest-html
    pytest-xdist
    pytest-cov
    colorlog

    # firemarshal
    gitpython
    humanfriendly
    doit

    # firesim
    argcomplete
    cryptography
    paramiko
    fab-classic
    boto3
    mypy-boto3-ec2
    mypy-boto3-s3
    colorama
    pylddwrap
    graphviz  # python-graphviz for topology diagrams
    aiohttp
  ]);
}
