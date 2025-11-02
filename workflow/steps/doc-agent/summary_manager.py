"""
SUMMARY.md manager
Responsible for parsing and updating mdBook's SUMMARY.md file
"""

import os
import re
from pathlib import Path


class SummaryManager:
    """SUMMARY.md manager"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.docs_base = self.project_root / "docs" / "bb-note" / "src"

    def parse_summary(self, summary_path):
        """Parse SUMMARY.md file and return structured data"""
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
        """Generate SUMMARY.md entry for new documentation"""
        docs_file = Path(docs_path)

        try:
            relative_path = docs_file.relative_to(self.docs_base)
        except ValueError:
            relative_path = docs_file

        title = Path(target_path).parts[-1] if Path(target_path).parts else "Unknown Document"
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
        """Determine indent level based on directory path and document type"""
        path_parts = Path(target_path).parts
        base_level = 1

        if doc_type == "rtl" and "scala" in path_parts:
            scala_index = path_parts.index("scala")
            base_level += len(path_parts) - scala_index - 1

        return max(0, base_level)

    def update_summary(self, summary_path, new_entry):
        """Update SUMMARY.md file, add new entry"""
        summary_data = self.parse_summary(summary_path)

        # Check for duplicates
        existing_paths = [entry["path"] for entry in summary_data["entries"]]
        if new_entry["path"] in existing_paths:
            return False, "Entry already exists"

        # Insert new entry
        lines = summary_data["original_content"].split("\n")
        new_lines = self._insert_entry(lines, new_entry)

        # Write back to file
        new_content = "\n".join(new_lines)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True, "SUMMARY.md updated"

    def _insert_entry(self, lines, new_entry):
        """Insert new entry at appropriate position"""
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
