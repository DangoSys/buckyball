"""文件操作相关的工具"""

import os
import json
from typing import Dict, Any
from .base import Tool


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
                    "description": "File path relative to work directory"
                }
            },
            "required": ["path"]
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
            return json.dumps({"error": "Cannot read file: not a text file or encoding issue"})


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
                    "description": "File path relative to work directory"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
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

            return json.dumps({
                "success": True,
                "path": file_path,
                "size": len(content)
            })

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
                    "description": "Directory path relative to work directory (default: .)"
                }
            }
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

            return json.dumps({
                "path": dir_path,
                "files": files,
                "count": len(files)
            })

        except Exception as e:
            return json.dumps({"error": f"Failed to list directory: {str(e)}"})
