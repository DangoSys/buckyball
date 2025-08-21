import asyncio
import os

def get_buckyball_path():
  current_dir = os.path.dirname(__file__)
  return os.path.dirname(os.path.dirname(current_dir))