import asyncio

config = {
  "type": "event",
  "name": "MonitorInit",
  "description": "run monitor-init.sh",
  "subscribes": ["install-monitor"],
  "emits": [],
  "input": { "type": "object", "properties": {} },
  "flows": ["install"],
}

async def handler(input, context):
  context.logger.info('Install – running monitor-init.sh', {})

  proc = await asyncio.create_subprocess_exec(
    '/bin/bash',
    '-lc',
    'bash "$PWD/steps/install/monitor-init.sh"',
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  stdout, stderr = await proc.communicate()
  if proc.returncode != 0:
    context.logger.error('monitor-init.sh failed', { 'stderr': (stderr or b'').decode('utf-8') })
    raise RuntimeError('monitor-init.sh failed')
  context.logger.info('monitor-init.sh output', { 'stdout': (stdout or b'').decode('utf-8') })

