import asyncio
import shlex

async def run(command_str, context, check=True):
  """
  Execute a command string and return the result.
  
  Args:
    command_str: The complete command string to execute
    context: The context object for logging
    check: If True, raise RuntimeError on non-zero return code
  
  Returns:
    tuple: (returncode, stdout, stderr)
  """
  context.logger.info(f'Running command: {command_str}', {})
  
  # Parse the command string into command and arguments
  try:
    parts = shlex.split(command_str)
    if not parts:
      raise ValueError("Empty command string")
    command = parts[0]
    args = parts[1:]
  except Exception as e:
    raise ValueError(f"Invalid command string '{command_str}': {e}")
  
  proc = await asyncio.create_subprocess_exec(
    command,
    *args,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  
  stdout, stderr = await proc.communicate()
  returncode = proc.returncode
  
  # Log the output
  if stdout:
    for line in stdout.decode('utf-8', errors='replace').splitlines():
      if line.strip():
        context.logger.info(line.strip(), {})
  
  if stderr:
    for line in stderr.decode('utf-8', errors='replace').splitlines():
      if line.strip():
        context.logger.error(line.strip(), {})
  
  if check and returncode != 0:
    raise RuntimeError(f'Command "{command_str}" failed with return code {returncode}')
  
  return returncode, stdout, stderr 
  