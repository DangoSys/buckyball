{ pkgs }:
let
  # The pinned OpenTitan regtool imports pkg_resources.
  setuptools = pkgs.python3Packages.buildPythonPackage {
    pname = "setuptools";
    version = "78.1.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/90/99/158ad0609729111163fc1f674a5a42f2605371a4cf036d0441070e2f7455/setuptools-78.1.1-py3-none-any.whl";
      sha256 = "c3a9c4211ff4c309edb8b8c4f1cbfa7ae324c4ba9f91ff254e3d305b9fd54561";
    };
    doCheck = false;
  };
in
pkgs.mkShell {
  packages = [
    pkgs.bender
    pkgs.gnumake
    pkgs.tools.verilator
    (pkgs.python3.withPackages (ps: [ setuptools ps.mako ps.hjson ps.flatdict ps.pyyaml ps.tabulate ]))
  ];
}
