import asyncio
import os
import sys

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from utils import run

config = {
  "type": "event",
  "name": "InstallChipyard",
  "description": "run install-chipyard.sh",
  "subscribes": ["start"],
  "emits": ["install-chipyard-finished"],
  "input": { "type": "object", "properties": {} },
  "flows": ["build-setup"],
}

pwd_path = os.path.dirname(os.path.abspath(__file__))

async def handler(input, context):
  context.logger.info('Install – running install-chipyard.sh', {})
  # step should not do shutdown here; it only emits completion

  try:
    # Use the generic run function to execute the install script
    await run(f'bash "{pwd_path}/install-chipyard.sh"', context)
  finally:
    await context.emit({
      "topic": 'install-chipyard-finished',
      "data": {}
    })

