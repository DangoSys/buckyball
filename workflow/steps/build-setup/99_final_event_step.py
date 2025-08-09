import asyncio
import os
import urllib.request

config = {
  "type": "event",
  "name": "FinalEvent",
  "description": "shutdown bbdev via BBDEV_DONE_URL when flow finishes",
  "subscribes": ["install-chipyard-finished"],
  "emits": [],
  "input": { "type": "object", "properties": {} },
  "flows": ["build-setup"],
}


async def handler(input, context):
  done_url = os.environ.get('BBDEV_DONE_URL')
  if not done_url:
    return
  try:
    await asyncio.to_thread(urllib.request.urlopen, done_url, timeout=3)
  except Exception:
    # ignore errors so flow doesn't fail on shutdown handshake
    pass

