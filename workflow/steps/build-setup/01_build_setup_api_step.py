config = {
  "type": "api",
  "name": "build-setupAPI",
  "description": "trigger build-setup flow",

  "path": "/build-setup",
  "method": "POST",

  "emits": ["build-setup"],
  "virtualSubscribes": ["/build-setup"],
  "flows": ["build-setup"],
}

async def handler(req, context):
  context.logger.info('build-setup – API received', { "body": req.get("body") })

  await context.emit({
    "topic": 'build-setup',
    "data": req.get("body") or {},
  })

  return {
    "status": 200,
    "body": {
      "traceId": context.trace_id,
      "message": 'build-setup emitted'
    },
  }

