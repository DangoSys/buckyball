config = {
  "type": "api",
  "name": "option-installAPI",
  "description": "trigger option-install flow",

  "path": "/build-setup/option-install",
  "method": "POST",

  "emits": ["option-install"],
  "virtualSubscribes": ["/build-setup/option-install"],
  "flows": ["build-setup"],
}

async def handler(req, context):
  context.logger.info('option-install – API received', { "body": req.get("body") })

  await context.emit({
    "topic": 'option-install',
    "data": req.get("body") or {},
  })

  return {
    "status": 200,
    "body": {
      "traceId": context.trace_id,
      "message": 'option-install emitted'
    },
  } 