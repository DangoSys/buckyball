"""
文档生成工具函数
"""

import os
from pathlib import Path


def detect_doc_type(target_path):
    """根据目录路径自动检测文档类型"""
    path_str = str(Path(target_path).resolve()).replace("\\", "/")

    if "arch/src/main/scala" in path_str:
        return "rtl"
    elif "bb-tests" in path_str:
        if "/workloads/" in path_str or path_str.endswith("/workloads"):
            return "workloads"
        elif "/customext/" in path_str or path_str.endswith("/customext"):
            return "customext"
        elif "/sardine/" in path_str or path_str.endswith("/sardine"):
            return "sardine"
        elif "/uvbb/" in path_str or path_str.endswith("/uvbb"):
            return "uvbb"
        return "workloads"
    elif "/scripts/" in path_str or path_str.endswith("/scripts"):
        return "script"
    elif "/sims/" in path_str or path_str.endswith("/sims"):
        return "sim"
    elif "/workflow/" in path_str or path_str.endswith("/workflow"):
        return "workflow"

    return "script"


def load_prompt_template(doc_type, target_path):
    """加载并处理prompt模板"""
    template_path = f"workflow/prompts/doc/{doc_type}-doc.md"

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    template = template.replace("[`目录相对路径`]", target_path)
    template = template.replace("@[`目录相对路径`]", f"@{target_path}")

    return template


def prepare_update_mode_prompt(template, target_path, mode):
    """为更新模式准备prompt"""
    if mode != "update":
        return template

    existing_doc_path = os.path.join(target_path, "README.md")
    if not os.path.exists(existing_doc_path):
        return template

    with open(existing_doc_path, "r", encoding="utf-8") as f:
        existing_content = f.read()

    update_instruction = f"""

## 更新模式特殊指令

你正在更新现有文档。请注意：
1. 仔细分析现有文档内容，保留准确和有价值的信息
2. 识别并更新过时、不准确或不完整的部分
3. 保持文档的整体结构和风格一致性
4. 如果现有内容准确且完整，请保留它们

现有文档内容：
```markdown
{existing_content}
```
"""
    return template + update_instruction
