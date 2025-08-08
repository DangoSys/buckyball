import asyncio

config = {
  "type": "event",
  "name": "RunInit",
  "description": "run init-chipyard.sh",
  "subscribes": ["install-start"],
  "emits": ["install-monitor"],
  "input": { "type": "object", "properties": {} },
  "flows": ["install"],
}

async def handler(input, context):
  context.logger.info('Install – running init-chipyard.sh', {})

  try:
    proc = await asyncio.create_subprocess_exec(
      '/bin/bash',
      '-lc',
      'bash "$PWD/steps/install/init-chipyard.sh"',
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
      context.logger.error('init-chipyard.sh failed', { 'stderr': (stderr or b'').decode('utf-8') })
      raise RuntimeError('init-chipyard.sh failed')
    context.logger.info('init-chipyard.sh output', { 'stdout': (stdout or b'').decode('utf-8') })
  finally:
    await context.emit({
      "topic": 'install-monitor',
      "data": {}
    })

