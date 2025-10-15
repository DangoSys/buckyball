"""文件操作相关的工具"""

import os
import json
import shutil
from typing import Dict, Any
from .base import Tool


class MakeDirTool(Tool):
    """创建目录工具"""

    def get_name(self) -> str:
        return "make_dir"

    def get_description(self) -> str:
        return "Create a new directory (supports creating parent directories)"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"}
            },
            "required": ["path"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        path = arguments.get("path")

        if not os.path.isabs(path):
            path = os.path.join(context.work_dir, path)

        try:
            if os.path.exists(path):
                return json.dumps({"status": "exists", "path": path})

            os.makedirs(path, exist_ok=True)
            context.log_info(f"Created directory: {path}")
            return json.dumps({"status": "success", "path": path})

        except Exception as e:
            return json.dumps({"error": f"Failed to create directory: {str(e)}"})


class GetPathTool(Tool):
    """获取路径信息工具"""

    def get_name(self) -> str:
        return "get_path_info"

    def get_description(self) -> str:
        return "Get absolute path and check if path exists"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to check (optional, defaults to work_dir)",
                }
            },
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        path = arguments.get("path", ".")

        if not os.path.isabs(path):
            path = os.path.join(context.work_dir, path)

        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)

        info = {
            "absolute_path": abs_path,
            "exists": exists,
            "work_dir": context.work_dir,
        }

        if exists:
            info["is_file"] = os.path.isfile(abs_path)
            info["is_dir"] = os.path.isdir(abs_path)

        return json.dumps(info, indent=2)


class GrepFilesTool(Tool):
    """搜索文件内容工具"""

    def get_name(self) -> str:
        return "grep_files"

    def get_description(self) -> str:
        return "Search for text pattern in files"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text pattern to search"},
                "path": {"type": "string", "description": "Directory or file path"},
                "file_ext": {
                    "type": "string",
                    "description": "File extension filter (e.g., '.scala')",
                },
            },
            "required": ["pattern", "path"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        pattern = arguments.get("pattern")
        path = arguments.get("path")
        file_ext = arguments.get("file_ext")

        if not os.path.isabs(path):
            path = os.path.join(context.work_dir, path)

        try:
            results = []
            files_to_search = []

            if os.path.isfile(path):
                files_to_search = [path]
            else:
                for root, _, files in os.walk(path):
                    for file in files:
                        if file_ext and not file.endswith(file_ext):
                            continue
                        files_to_search.append(os.path.join(root, file))

            for filepath in files_to_search[:100]:  # 限制文件数
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern in line:
                                results.append(
                                    {
                                        "file": filepath,
                                        "line": line_num,
                                        "content": line.strip()[:150],
                                    }
                                )
                                if len(results) >= 50:  # 限制结果数
                                    break
                except Exception:
                    continue

                if len(results) >= 50:
                    break

            return json.dumps({"matches": len(results), "results": results}, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})


class DeleteFileTool(Tool):
    """删除文件工具"""

    def get_name(self) -> str:
        return "delete_file"

    def get_description(self) -> str:
        return "Delete a file (use with caution)"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file to delete"}
            },
            "required": ["path"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        path = arguments.get("path")

        if not os.path.isabs(path):
            path = os.path.join(context.work_dir, path)

        try:
            if not os.path.exists(path):
                return json.dumps({"status": "not_found", "path": path})

            if os.path.isfile(path):
                os.remove(path)
                context.log_info(f"Deleted file: {path}")
                return json.dumps({"status": "success", "path": path})
            else:
                return json.dumps({"error": "Path is a directory, not a file"})

        except Exception as e:
            return json.dumps({"error": f"Failed to delete: {str(e)}"})


class ReadFileTool(Tool):
    """读取文件内容"""

    def get_name(self) -> str:
        return "read_file"

    def get_description(self) -> str:
        return "Read the content of a file"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to work directory",
                }
            },
            "required": ["path"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        file_path = arguments.get("path")

        if not file_path:
            return json.dumps({"error": "Missing required parameter: path"})

        full_path = os.path.join(context.work_dir, file_path)

        # 安全检查：防止路径穿越
        abs_full = os.path.abspath(full_path)
        abs_work = os.path.abspath(context.work_dir)
        if not abs_full.startswith(abs_work):
            return json.dumps({"error": "Access denied: path outside work directory"})

        if not os.path.exists(full_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        if not os.path.isfile(full_path):
            return json.dumps({"error": f"Not a file: {file_path}"})

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            context.log_info(f"Tool: read_file({file_path}) - {len(content)} chars")
            return content

        except UnicodeDecodeError:
            return json.dumps(
                {"error": "Cannot read file: not a text file or encoding issue"}
            )


class WriteFileTool(Tool):
    """写入文件内容"""

    def get_name(self) -> str:
        return "write_file"

    def get_description(self) -> str:
        return "Write content to a file (creates directories if needed)"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to work directory",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        file_path = arguments.get("path")
        content = arguments.get("content")

        if not file_path:
            return json.dumps({"error": "Missing required parameter: path"})

        if content is None:
            return json.dumps({"error": "Missing required parameter: content"})

        full_path = os.path.join(context.work_dir, file_path)

        # 安全检查：防止路径穿越
        abs_full = os.path.abspath(full_path)
        abs_work = os.path.abspath(context.work_dir)
        if not abs_full.startswith(abs_work):
            return json.dumps({"error": "Access denied: path outside work directory"})

        try:
            # 创建目录
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # 写入文件
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            context.log_info(f"Tool: write_file({file_path}) - {len(content)} chars")

            return json.dumps(
                {"success": True, "path": file_path, "size": len(content)}
            )

        except Exception as e:
            return json.dumps({"error": f"Failed to write file: {str(e)}"})


class ListFilesTool(Tool):
    """列出目录中的文件"""

    def get_name(self) -> str:
        return "list_files"

    def get_description(self) -> str:
        return "List files in a directory"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to work directory (default: .)",
                }
            },
        }

    def execute(self, arguments: Dict[str, Any], context: Any) -> str:
        dir_path = arguments.get("path", ".")
        full_path = os.path.join(context.work_dir, dir_path)

        # 安全检查
        abs_full = os.path.abspath(full_path)
        abs_work = os.path.abspath(context.work_dir)
        if not abs_full.startswith(abs_work):
            return json.dumps({"error": "Access denied: path outside work directory"})

        if not os.path.exists(full_path):
            return json.dumps({"error": f"Directory not found: {dir_path}"})

        if not os.path.isdir(full_path):
            return json.dumps({"error": f"Not a directory: {dir_path}"})

        try:
            files = os.listdir(full_path)
            context.log_info(f"Tool: list_files({dir_path}) - {len(files)} items")

            return json.dumps({"path": dir_path, "files": files, "count": len(files)})

        except Exception as e:
            return json.dumps({"error": f"Failed to list directory: {str(e)}"})
