import subprocess


def run_bbdev_case(command: str):
  result = subprocess.run(command, shell=True)
  if result.returncode != 0:
    raise RuntimeError(f"command failed({result.returncode}): {command}")
