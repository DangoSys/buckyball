config = {
  "type": "noop",
  "name": "Optional Install Anaconda",
  "description": "Install Tools if the user wants to",

  # Used mostly to connect nodes that emit to this
  "virtualSubscribes": [],

  # Used mostly to connect nodes that subscribes to this
  "virtualEmits": ["/build-setup/option-install"],

  # The flows this step belongs to, will be available in Workbench
  "flows": ["build-setup"],
}