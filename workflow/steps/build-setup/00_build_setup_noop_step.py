config = {
  "type": "noop",
  "name": "Init Chipyard Trigger",
  "description": "Trigger the Chipyard initialization process",

  # Used mostly to connect nodes that emit to this
  "virtualSubscribes": [],

  # Used mostly to connect nodes that subscribes to this
  "virtualEmits": ["/build-setup"],

  # The flows this step belongs to, will be available in Workbench
  "flows": ["build-setup"],
}