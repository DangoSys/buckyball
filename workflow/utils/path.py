import asyncio
import os

def get_buckyball_path(context):
  current_dir = os.path.dirname(__file__)
  root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
  return os.path.join(root_dir, 'tools', 'buckyball')