import asyncio
import os
import urllib.request
import urllib.error

config = {
  "type": "event",
  "name": "InstallChipyard",
  "description": "run install-chipyard.sh",
  "subscribes": ["build-setup"],
  "emits": ["install-chipyard-finished"],
  "input": { "type": "object", "properties": {} },
  "flows": ["build-setup"],
}

pwd_path = os.path.dirname(os.path.abspath(__file__))

async def handler(input, context):
  context.logger.info('Install – running install-chipyard.sh', {})
  # step should not do shutdown here; it only emits completion

  async def _relay(stream: asyncio.StreamReader, log_fn, label: str) -> None:
    while True:
      line = await stream.readline()
      if not line:
        break
      text = line.decode('utf-8', errors='replace').rstrip('\n')
      log_fn(text)
      # log_fn(f'install-chipyard.sh {label}', { 'line': text })

  try:
    proc = await asyncio.create_subprocess_exec(
      '/bin/bash',
      '-lic',
      f'bash "{pwd_path}/install-chipyard.sh"',
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )

    tasks = [
      asyncio.create_task(_relay(proc.stdout, context.logger.info, 'stdout')),
      asyncio.create_task(_relay(proc.stderr, context.logger.error, 'stderr')),
    ]

    returncode = await proc.wait()
    await asyncio.gather(*tasks, return_exceptions=True)

    if returncode != 0:
      raise RuntimeError('install-chipyard.sh failed')
  finally:
    await context.emit({
      "topic": 'install-chipyard-finished',
      "data": {}
    })

