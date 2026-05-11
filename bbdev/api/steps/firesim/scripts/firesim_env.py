"""FireSim environment setup."""

import os
import subprocess


def setup_firesim_env() -> dict:
    """Setup FireSim env vars + SSH agent. Returns env dict for subprocess."""
    env = os.environ.copy()
    env["FIRESIM_SOURCED"] = "1"
    env["FIRESIM_RUNFARM_PREFIX"] = ""

    key = os.path.expanduser("~/.ssh/firesim.pem")
    if os.path.exists(key) and "firesim.pem" not in subprocess.run(
        ["ssh-add", "-l"], capture_output=True, text=True
    ).stdout:
        if not env.get("SSH_AUTH_SOCK"):
            for line in subprocess.run(
                ["ssh-agent", "-s"], capture_output=True, text=True
            ).stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v.split(";", 1)[0]
        subprocess.run(["ssh-add", key], env=env, capture_output=True)

    return env
