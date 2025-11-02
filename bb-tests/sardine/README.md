# Sardine Test Framework

A simple test management framework based on pytest, specifically designed for executing script tests.

## Directory Structure

```
sardine/
├── pytest.ini         # pytest configuration
├── conftest.py         # pytest fixture configuration
├── tests/              # test directory
│   └── test_basic.py   # basic test cases
└── README.md           # documentation
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Tests

<!-- ### Run All Tests
```bash
cd voyager-test/scripts/sardine
pytest
``` -->

### Quick Start
```bash
# Run smoke tests
python -m pytest -m smoke -s -v -n auto
```

### Run Specific Test File
```bash
pytest tests/test_basic.py
```

### Run Specific Test Function
```bash
pytest tests/test_basic.py::test_verilator_help
```

### Run Tests in Parallel
```bash
# Automatically detect CPU cores for parallel execution
pytest -n auto

# Specify 4 parallel processes
pytest -n 4

# Run specific marked tests in parallel
pytest -m smoke -n auto

# Run specific file in parallel
pytest tests/test_basic.py -n auto
```

## Test Markers

- `smoke`: Quick smoke tests
- `verilator`: Verilator simulation tests
- `vcs`: VCS simulation tests
- `unit`: Unit tests
- `integration`: Integration tests

## Adding New Tests

Create a new test file under the `tests/` directory using the following template:

```python
import pytest

@pytest.mark.smoke
def test_your_script(script_runner):
  """Test your script."""
  result = script_runner("your-script.sh", ["arg1", "arg2"], timeout=60)

  # Verify results
  assert result["success"], f"Script failed: {result['stderr']}"
  assert "expected_output" in result["stdout"]
```

## Test Result Storage

Test results are automatically saved to the `reports/` directory:

```
reports/
├── report.html      # HTML test report (recommended for viewing)
├── junit.xml        # JUnit XML format report
└── test_output.log  # Complete test output log
```

### Viewing Test Results

1. **HTML Report** (Recommended): Open `http://server-ip:3000` in browser to view the latest report, full report located at http://server-ip:3000/{commit}, e.g. http://server-ip:3000/f34ddb98
2. **Console Output**: Directly displayed in terminal (including all print statements)
3. **XML Report**: `reports/junit.xml` (for CI integration)
