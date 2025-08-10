import asyncio
import os
import sys

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from utils import run

config = {
  "type": "event",
  "name": "InstallAnaconda",
  "description": "Install latest Anaconda to tools directory",
  "subscribes": ["option-install"],
  "emits": ["install-anaconda-finished"],
  "input": { "type": "object", "properties": {} },
  "flows": ["build-setup"],
}

async def handler(input, context):
  # Check if install_anaconda is True in the input data
  install_anaconda = input.get("install_anaconda", False)
  
  if not install_anaconda:
    context.logger.info('InstallAnaconda – Skipping installation (install_anaconda is False)', {})
    await context.emit({
      "topic": 'install-anaconda-finished',
      "data": { "installed": False, "reason": "install_anaconda is False" }
    })
    return
  
  context.logger.info('InstallAnaconda – Starting Anaconda installation', {})
  
  try:
    # Create tools directory if it doesn't exist
    # Navigate from current file: workflow/steps/build-setup/tools/00_install_anaconda_step.py
    # to tools directory: ../../../../tools (4 levels up from workflow/steps/build-setup/tools)
    current_dir = os.path.dirname(__file__)  # workflow/steps/build-setup/tools
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))  # Go up 4 levels to get to root
    tools_dir = os.path.join(root_dir, 'tools')
    os.makedirs(tools_dir, exist_ok=True)
    
    # Check if Anaconda is already installed
    anaconda_path = os.path.join(tools_dir, 'anaconda3')
    if os.path.exists(anaconda_path):
      context.logger.info('InstallAnaconda – Anaconda already exists at tools/anaconda3', {})
      await context.emit({
        "topic": 'install-anaconda-finished',
        "data": { "installed": True, "reason": "already exists", "path": anaconda_path }
      })
      return
    
    # Download latest Anaconda installer
    context.logger.info('InstallAnaconda – Downloading latest Anaconda installer', {})
    
    # Get the latest Anaconda download URL (this is a simplified approach)
    # In a real implementation, you might want to scrape the Anaconda website or use their API
    download_url = "https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh"
    installer_path = os.path.join(tools_dir, "anaconda_installer.sh")
    
    # Download the installer
    await run(f'wget -O {installer_path} {download_url}', context)
    
    # Make the installer executable
    await run(f'chmod +x {installer_path}', context)
    
    # Install Anaconda to tools directory
    context.logger.info('InstallAnaconda – Installing Anaconda to tools directory', {})
    
    # Use batch mode installation to tools/anaconda3
    install_cmd = f'bash {installer_path} -b -p {anaconda_path}'
    await run(install_cmd, context)
    
    # Clean up the installer
    os.remove(installer_path)
    
    context.logger.info('InstallAnaconda – Successfully installed Anaconda', { "path": anaconda_path })
    
    await context.emit({
      "topic": 'install-anaconda-finished',
      "data": { "installed": True, "path": anaconda_path }
    })
    
  except Exception as e:
    context.logger.error(f'InstallAnaconda – Error installing Anaconda: {str(e)}', {})
    await context.emit({
      "topic": 'install-anaconda-finished',
      "data": { "installed": False, "error": str(e) }
    })
    raise 