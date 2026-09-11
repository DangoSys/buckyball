import os
from typing import Optional, List


def search_workload(search_dir: str, filename: str) -> Optional[str]:
    """
    Recursively resolve one workload file name in the directory and its subdirectories.

    Args:
      search_dir: Root directory to search
      filename: Filename to search for

    Returns:
      Absolute path of the file, or None if not found.

    Raises:
      ValueError: the file name has more than one match.
    """
    if os.path.isfile(filename):
        return os.path.abspath(filename)

    matches = search_workload_all(search_dir, filename)
    if len(matches) > 1:
        raise ValueError(
            f"workload file name {filename!r} is ambiguous under {search_dir}: "
            + ", ".join(matches)
        )
    return matches[0] if matches else None


def search_workload_all(search_dir: str, filename: str) -> List[str]:
    """
    Recursively search for a specified filename in the directory and its subdirectories, returning all matches

    Args:
      search_dir: Root directory to search
      filename: Filename to search for

    Returns:
      List of absolute paths of all found files
    """
    results = []

    if not os.path.exists(search_dir):
        return results

    for root, dirs, files in os.walk(search_dir):
        dirs.sort()
        if filename in files:
            results.append(os.path.abspath(os.path.join(root, filename)))

    return results


def search_workload_pattern(search_dir: str, pattern: str) -> List[str]:
    """
    Recursively search for files matching a specified pattern in the directory and its subdirectories

    Args:
      search_dir: Root directory to search
      pattern: Filename pattern (supports wildcards * and ?)

    Returns:
      List of absolute paths of all matching files
    """
    import fnmatch

    results = []

    if not os.path.exists(search_dir):
        return results

    for root, dirs, files in os.walk(search_dir):
        dirs.sort()
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                results.append(os.path.abspath(os.path.join(root, file)))

    return results
