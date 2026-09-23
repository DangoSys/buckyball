import shlex
import subprocess


def run_bbdev_case(command: str):
  subprocess.run(shlex.split(command), check=True)
