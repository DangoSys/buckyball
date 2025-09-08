import os
from typing import Optional, List


def search_workload(search_dir: str, filename: str) -> Optional[str]:
  """
  在指定文件夹及其子文件夹下递归搜索指定文件名
  
  Args:
    search_dir: 搜索的根目录
    filename: 要搜索的文件名
    
  Returns:
    找到的文件的绝对路径，如果未找到则返回None
  """
  if not os.path.exists(search_dir):
    return None
    
  for root, dirs, files in os.walk(search_dir):
    if filename in files:
      return os.path.abspath(os.path.join(root, filename))
  
  return None


def search_workload_all(search_dir: str, filename: str) -> List[str]:
  """
  在指定文件夹及其子文件夹下递归搜索指定文件名，返回所有匹配的文件
  
  Args:
    search_dir: 搜索的根目录
    filename: 要搜索的文件名
    
  Returns:
    所有找到的文件的绝对路径列表
  """
  results = []
  
  if not os.path.exists(search_dir):
    return results
    
  for root, dirs, files in os.walk(search_dir):
    if filename in files:
      results.append(os.path.abspath(os.path.join(root, filename)))
  
  return results


def search_workload_pattern(search_dir: str, pattern: str) -> List[str]:
  """
  在指定文件夹及其子文件夹下递归搜索匹配指定模式的文件
  
  Args:
    search_dir: 搜索的根目录
    pattern: 文件名模式 (支持通配符 * 和 ?)
    
  Returns:
    所有匹配的文件的绝对路径列表
  """
  import fnmatch
  results = []
  
  if not os.path.exists(search_dir):
    return results
    
  for root, dirs, files in os.walk(search_dir):
    for file in files:
      if fnmatch.fnmatch(file, pattern):
        results.append(os.path.abspath(os.path.join(root, file)))
  
  return results