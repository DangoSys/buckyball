{ lib, stdenv, makeWrapper, python312, nodejs_22, nodePackages }:

let
  pythonEnv = python312.withPackages (ps: with ps; [
    pydantic
    requests
  ]);
in
stdenv.mkDerivation {
  pname = "bbdev";
  version = "0.1.0";

  src = ./..;

  nativeBuildInputs = [ makeWrapper ];

  dontBuild = true;

  installPhase = ''
    runHook preInstall

    # Install the api directory (Motia backend + utils + steps)
    mkdir -p $out/lib/bbdev
    cp -r api/* $out/lib/bbdev/

    # Install the bbdev CLI script
    mkdir -p $out/bin
    cp bbdev $out/bin/.bbdev-unwrapped
    chmod +x $out/bin/.bbdev-unwrapped

    # Patch workflow_dir to use BBDEV_API_DIR env var if set,
    # otherwise fall back to bbdev/api relative to CWD (project root)
    substituteInPlace $out/bin/.bbdev-unwrapped \
      --replace 'workflow_dir = os.path.dirname(os.path.abspath(__file__))' \
                'workflow_dir = os.environ.get("BBDEV_API_DIR", os.path.join(os.getcwd(), "bbdev", "api"))'

    # Wrap bbdev with Python env and Node.js in PATH
    makeWrapper $out/bin/.bbdev-unwrapped $out/bin/bbdev \
      --prefix PATH : ${lib.makeBinPath [ pythonEnv nodejs_22 nodePackages.pnpm ]} \
      --prefix PYTHONPATH : $out/lib/bbdev

    runHook postInstall
  '';

  meta = with lib; {
    description = "Agent-friendly developer toolchain backend powered by Motia";
    license = licenses.mit;
    mainProgram = "bbdev";
  };
}
