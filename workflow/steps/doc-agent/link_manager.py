"""
符号链接管理器
负责创建和验证文档目录结构中的符号链接
"""

import os
import shutil
from pathlib import Path


class LinkManager:
    """符号链接管理器"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.docs_base = self.project_root / "docs" / "bb-note" / "src"

    def create_docs_structure(self, target_path):
        """创建对应的文档目录结构"""
        docs_path = self._convert_to_docs_path(target_path)
        docs_dir = Path(docs_path).parent
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_path

    def _convert_to_docs_path(self, target_path):
        """将代码目录路径转换为对应的文档目录路径"""
        target_path = Path(target_path).resolve()

        try:
            relative_path = target_path.relative_to(self.project_root)
        except ValueError:
            relative_path = (
                Path(*target_path.parts[-3:])
                if len(target_path.parts) >= 3
                else target_path
            )

        docs_path = self.docs_base / relative_path / "README.md"
        return str(docs_path)

    def create_symbolic_link(self, source_path, target_path):
        """创建符号链接"""
        source = Path(source_path)
        target = Path(target_path)

        if not source.exists():
            raise FileNotFoundError(f"源文件不存在: {source_path}")

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() or target.is_symlink():
            target.unlink()

        try:
            relative_source = os.path.relpath(source, target.parent)
            target.symlink_to(relative_source)
            return True
        except Exception as e:
            try:
                shutil.copy2(source, target)
                return True
            except Exception as copy_error:
                raise Exception(
                    f"创建符号链接和复制文件都失败: {str(e)}, {str(copy_error)}"
                )

    def validate_links(self, docs_base_path=None):
        """验证符号链接的有效性"""
        if docs_base_path is None:
            docs_base_path = self.docs_base
        else:
            docs_base_path = Path(docs_base_path)

        invalid_links = []
        valid_links = []

        for link_path in docs_base_path.rglob("*"):
            if link_path.is_symlink():
                try:
                    if link_path.exists():
                        valid_links.append(str(link_path))
                    else:
                        invalid_links.append(
                            {
                                "link": str(link_path),
                                "target": str(link_path.readlink()),
                                "error": "目标文件不存在",
                            }
                        )
                except Exception as e:
                    invalid_links.append(
                        {"link": str(link_path), "target": "无法读取", "error": str(e)}
                    )

        return {
            "valid_links": valid_links,
            "invalid_links": invalid_links,
            "total_links": len(valid_links) + len(invalid_links),
        }
