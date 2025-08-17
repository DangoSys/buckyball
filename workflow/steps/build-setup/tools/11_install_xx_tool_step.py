import asyncio
import os
import sys

# Add the utils directory to the Python path  
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils import run

config = {
  "type": "event",
  "name": "InstallXXTool",
  "description": "Install XX tool to tools directory",
  "subscribes": ["option-install"],
  "emits": ["install-xx-tool-finished"],
  "input": { "type": "object", "properties": {} },
  "flows": ["build-setup"],
}

async def handler(input, context):
  # Check if install_xx is True in the input data
  install_xx = input.get("install_xx", False)
  
  if not install_xx:
    context.logger.info('InstallXXTool – Skipping installation (install_xx is False)', {})
    await context.emit({
      "topic": 'install-xx-tool-finished',
      "data": { "installed": False, "reason": "install_xx is False" }
    })
    return
  
  context.logger.info('InstallXXTool – Starting XX tool installation', {})
  
  try:
    # Create tools directory if it doesn't exist
    # Navigate from current file: workflow/steps/build-setup/tools/01_install_xx_tool_step.py
    # to tools directory: ../../../../tools (4 levels up from workflow/steps/build-setup/tools)
    current_dir = os.path.dirname(__file__)  # workflow/steps/build-setup/tools
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))  # Go up 4 levels to get to root
    tools_dir = os.path.join(root_dir, 'tools')
    os.makedirs(tools_dir, exist_ok=True)
    
    # Check if XX tool is already installed
    xx_tool_path = os.path.join(tools_dir, 'xx-tool')
    if os.path.exists(xx_tool_path):
      context.logger.info('InstallXXTool – XX tool already exists at tools/xx-tool', {})
      await context.emit({
        "topic": 'install-xx-tool-finished',
        "data": { "installed": True, "reason": "already exists", "path": xx_tool_path }
      })
      return
    
    # Download XX tool
    context.logger.info('InstallXXTool – Downloading XX tool', {})
    
    # Example: Download XX tool (replace with actual download URL)
    download_url = "https://example.com/xx-tool.tar.gz"
    download_path = os.path.join(tools_dir, "xx-tool.tar.gz")
    
    # Download the tool
    await run(f'wget -O {download_path} {download_url}', context)
    
    # Extract the tool
    context.logger.info('InstallXXTool – Extracting XX tool', {})
    await run(f'tar -xzf {download_path} -C {tools_dir}', context)
    
    # Clean up the downloaded file
    os.remove(download_path)
    
    context.logger.info('InstallXXTool – Successfully installed XX tool', { "path": xx_tool_path })
    
    await context.emit({
      "topic": 'install-xx-tool-finished',
      "data": { "installed": True, "path": xx_tool_path }
    })
    
  except Exception as e:
    context.logger.error(f'InstallXXTool – Error installing XX tool: {str(e)}', {})
    await context.emit({
      "topic": 'install-xx-tool-finished',
      "data": { "installed": False, "error": str(e) }
    })
    raise 