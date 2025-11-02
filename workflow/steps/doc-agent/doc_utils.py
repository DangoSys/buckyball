"""
Documentation generation utility functions
"""

import os
from pathlib import Path


def detect_doc_type(target_path):
    """Automatically detect document type based on directory path"""
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
    """Load and process prompt template"""
    template_path = f"workflow/prompts/doc/{doc_type}-doc.md"

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    template = template.replace("[`目录相对路径`]", target_path)
    template = template.replace("@[`目录相对路径`]", f"@{target_path}")

    return template


def prepare_update_mode_prompt(template, target_path, mode):
    """Prepare prompt for update mode"""
    if mode != "update":
        return template

    existing_doc_path = os.path.join(target_path, "README.md")
    if not os.path.exists(existing_doc_path):
        return template

    with open(existing_doc_path, "r", encoding="utf-8") as f:
        existing_content = f.read()

    update_instruction = f"""

## Special Instructions for Update Mode

You are updating existing documentation. Please note:
1. Carefully analyze existing documentation content, retain accurate and valuable information
2. Identify and update outdated, inaccurate or incomplete sections
3. Maintain overall document structure and style consistency
4. If existing content is accurate and complete, retain it

Existing documentation content:
```markdown
{existing_content}
```
"""
    return template + update_instruction
