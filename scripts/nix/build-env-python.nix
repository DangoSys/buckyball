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
in
{
  # Python and pip packages
  python3 = pkgs.python3;

  # Python packages
  python3Packages = pkgs.python3.withPackages (ps: with ps; [
    # bbdev
    pydantic
    python-dotenv
    httpx
    mcp
    redis
    httpx-sse
    requests
    pysocks
    allure-pytest
    matplotlib

    # pre-commit hooks (language: system use)
    black
    flake8
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
