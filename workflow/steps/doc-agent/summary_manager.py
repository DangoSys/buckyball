"""
SUMMARY.md管理器
负责解析和更新mdBook的SUMMARY.md文件
"""

import os
import re
from pathlib import Path


class SummaryManager:
    """SUMMARY.md管理器"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.docs_base = self.project_root / "docs" / "bb-note" / "src"

    def parse_summary(self, summary_path):
        """解析SUMMARY.md文件，返回结构化数据"""
        if not os.path.exists(summary_path):
            return {"sections": [], "entries": [], "original_content": ""}

        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        entries = []

        for line in lines:
            line = line.rstrip()
            if line.strip().startswith("-"):
                match = re.match(r"\s*-\s*\[([^\]]+)\]\(([^)]+)\)", line)
                if match:
                    title, path = match.groups()
                    entries.append({"title": title, "path": path, "line": line})

        return {"entries": entries, "original_content": content}

    def generate_entry(self, target_path, docs_path, doc_type):
        """为新文档生成SUMMARY.md条目"""
        docs_file = Path(docs_path)

        try:
            relative_path = docs_file.relative_to(self.docs_base)
        except ValueError:
            relative_path = docs_file

        title = Path(target_path).parts[-1] if Path(target_path).parts else "未知文档"
        indent_level = self._determine_indent_level(target_path, doc_type)
        indent = "\t" * indent_level

        entry = f"{indent}- [{title}](./{relative_path})"

        return {
            "title": title,
            "path": f"./{relative_path}",
            "line": entry,
            "target_path": target_path,
            "doc_type": doc_type,
        }

    def _determine_indent_level(self, target_path, doc_type):
        """根据目录路径和文档类型确定缩进级别"""
        path_parts = Path(target_path).parts
        base_level = 1

        if doc_type == "rtl" and "scala" in path_parts:
            scala_index = path_parts.index("scala")
            base_level += len(path_parts) - scala_index - 1

        return max(0, base_level)

    def update_summary(self, summary_path, new_entry):
        """更新SUMMARY.md文件，添加新条目"""
        summary_data = self.parse_summary(summary_path)

        # 检查重复
        existing_paths = [entry["path"] for entry in summary_data["entries"]]
        if new_entry["path"] in existing_paths:
            return False, "条目已存在"

        # 插入新条目
        lines = summary_data["original_content"].split("\n")
        new_lines = self._insert_entry(lines, new_entry)

        # 写回文件
        new_content = "\n".join(new_lines)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True, "SUMMARY.md已更新"

    def _insert_entry(self, lines, new_entry):
        """在适当位置插入新条目"""
        new_lines = []
        inserted = False

        for line in lines:
            new_lines.append(line)
            if "contributors" in line.lower() and not inserted:
                new_lines.insert(-1, new_entry["line"])
                inserted = True

        if not inserted:
            new_lines.append(new_entry["line"])

        return new_lines
