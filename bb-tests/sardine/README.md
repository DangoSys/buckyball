# Sardine Test Framework

基于 pytest 的简单测试管理框架，专门用于执行脚本测试。

## 目录结构

```
sardine/
├── pytest.ini         # pytest 配置
├── conftest.py         # pytest 夹具配置
├── tests/              # 测试目录
│   └── test_basic.py   # 基本测试用例
└── README.md           # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

<!-- ### 运行所有测试
```bash
cd voyager-test/scripts/sardine
pytest
``` -->

### Quick Start
```bash
# 运行 smoke 测试
python -m pytest -m smoke -s -v -n auto
```

### 运行特定测试文件
```bash
pytest tests/test_basic.py
```

### 运行特定测试函数
```bash
pytest tests/test_basic.py::test_verilator_help
```

### 并行运行测试
```bash
# 自动检测CPU核心数并行运行
pytest -n auto

# 指定4个并行进程
pytest -n 4

# 并行运行特定标记的测试
pytest -m smoke -n auto

# 并行运行特定文件
pytest tests/test_basic.py -n auto
```

## 测试标记

- `smoke`: 快速烟雾测试
- `verilator`: Verilator 仿真测试
- `vcs`: VCS 仿真测试
- `unit`: 单元测试
- `integration`: 集成测试

## 添加新测试

在 `tests/` 目录下创建新的测试文件，使用以下模板：

```python
import pytest

@pytest.mark.smoke
def test_your_script(script_runner):
  """Test your script."""
  result = script_runner("your-script.sh", ["arg1", "arg2"], timeout=60)
  
  # 验证结果
  assert result["success"], f"Script failed: {result['stderr']}"
  assert "expected_output" in result["stdout"]
```

## 测试结果存储

测试结果会自动保存到 `reports/` 目录：

```
reports/
├── report.html      # HTML测试报告（推荐查看）
├── junit.xml        # JUnit XML格式报告
└── test_output.log  # 完整测试输出日志
```

### 查看测试结果

1. **HTML报告**（推荐）: 在浏览器中打开 `http://服务器ip:3000` 查看最近一次报告，完整报告位于 http://服务器ip:3000/{commit}, 例如 http://服务器ip:3000/f34ddb98
2. **控制台输出**: 直接显示在终端（包括所有print语句）
3. **XML报告**: `reports/junit.xml` （用于CI集成）
