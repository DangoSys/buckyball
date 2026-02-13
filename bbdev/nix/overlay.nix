# Motia Python dependencies overlay
# Provides Python packages needed by the project
final: prev: {
  bbdevPythonPkgs = final.python312.withPackages (ps: with ps; [
    pydantic
    requests
  ]);
}
