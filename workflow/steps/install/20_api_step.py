config = {
  "type": "api",
  "name": "InstallAPI",
  "description": "trigger install flow",

  "path": "/install",
  "method": "POST",

  "emits": ["install-start"],
  "virtualSubscribes": ["/install"],
  "flows": ["install"],
}

async def handler(req, context):
  context.logger.info('Install – API received', { "body": req.get("body") })

  await context.emit({
    "topic": 'install-start',
    "data": req.get("body") or {},
  })

  return {
    "status": 200,
    "body": {
      "traceId": context.trace_id,
      "message": 'install-start emitted'
    },
  }

