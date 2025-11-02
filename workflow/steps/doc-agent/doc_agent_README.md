# Doc-Agent User Documentation

## Overview

Doc-Agent is an automated documentation generation system built on the Motia framework, capable of automatically generating high-quality Chinese technical documentation for different types of directories in the codebase. The system supports various document types, including RTL hardware documentation, test documentation, script documentation, simulator documentation, and workflow documentation.

## System Architecture

Doc-Agent adopts an event-driven microservice architecture:

```
API Interface → Event Processing → Document Generation → Integration Management → mdBook Integration
```

- **API Step**: Receives HTTP requests and triggers document generation events
- **Event Step**: Handles document generation logic and calls LLM API
- **Integration Step**: Manages symbolic links and SUMMARY.md updates
- **Template System**: Provides dedicated templates for various document types

## API Interface Description

### Endpoint Information

- **URL**: `POST /doc/generate`
- **Content-Type**: `application/json`
- **Description**: Generate documentation for specified directory

### Request Parameters

| Parameter | Type | Required | Description |
|--------|------|------|------|
| `target_path` | string | Yes | Relative path to target code directory |
| `mode` | string | Yes | Generation mode: `create` or `update` |

#### Mode Description

- **create**: Create new documentation from scratch, suitable for directories without existing documentation
- **update**: Update existing documentation, retain accurate content, correct outdated information

### Response Format

#### Success Response (200)
```json
{
  "status": "success",
  "message": "Documentation generation task started",
  "data": {
    "target_path": "arch/src/main/scala/framework",
    "mode": "create",
    "doc_type": "rtl",
    "trace_id": "doc-gen-20241201-001"
  }
}
```

#### Error Response (400/500)
```json
{
  "status": "error",
  "message": "Error description",
  "error_code": "INVALID_PATH",
  "details": {
    "target_path": "Provided path does not exist or is inaccessible"
  }
}
```

## Usage Examples

### Basic Usage

#### 1. Generate RTL Hardware Documentation
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework/builtin",
    "mode": "create"
  }'
```

#### 2. Update Test Documentation
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "bb-tests/workloads/src",
    "mode": "update"
  }'
```

#### 3. Generate Script Documentation
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "scripts/docker",
    "mode": "create"
  }'
```

### Batch Processing Examples

#### Process Entire Test Directory
```bash
# Process all bb-tests subdirectories
for dir in workloads customext sardine uvbb; do
  curl -X POST http://localhost:8080/doc/generate \
    -H "Content-Type: application/json" \
    -d "{\"target_path\": \"bb-tests/$dir\", \"mode\": \"create\"}"
  sleep 2  # Avoid concurrent overload
done
```

#### Batch Update Existing Documentation
```bash
# Update documentation for all major directories
targets=("arch/src/main/scala" "bb-tests/workloads" "scripts" "sims/func-sim" "workflow/steps")

for target in "${targets[@]}"; do
  echo "Updating documentation: $target"
  curl -X POST http://localhost:8080/doc/generate \
    -H "Content-Type: application/json" \
    -d "{\"target_path\": \"$target\", \"mode\": \"update\"}"
  echo "Waiting for processing to complete..."
  sleep 5
done
```

## Supported Document Types

The system automatically identifies document types based on directory paths:

| Path Pattern | Doc Type | Template File | Description |
|----------|----------|----------|------|
| `arch/src/main/scala/**` | RTL | rtl-doc.md | RTL hardware module documentation |
| `bb-tests/workloads/**` | Workloads | workloads-doc.md | Workload test documentation |
| `bb-tests/customext/**` | CustomExt | customext-doc.md | Custom extension test documentation |
| `bb-tests/sardine/**` | Sardine | sardine-doc.md | Sardine test framework documentation |
| `bb-tests/uvbb/**` | UVBB | uvbb-doc.md | UVBB test documentation |
| `scripts/**` | Script | script-doc.md | Script and tool documentation |
| `sims/**` | Simulator | sim-doc.md | Simulator documentation |
| `workflow/**` | Workflow | workflow-doc.md | Workflow and automation documentation |

## Documentation Standards

All generated documentation follows unified standards:

### Language Specifications
- **Main Language**: Chinese
- **Technical Terms**: Keep original English
- **Code Comments**: Provide Chinese explanations
- **Professional Tone**: Avoid using emojis and informal expressions

### Format Specifications
- **Markdown Format**: Standard Markdown syntax
- **Code Blocks**: Use syntax highlighting
- **Links**: Use relative paths
- **Diagrams**: Support Mermaid diagrams

### Structure Specifications
Different document types have different structure requirements, but all include:
- Overview section
- Code structure analysis
- Detailed explanation
- Usage examples (if applicable)

## Integration Features

### Automatic Integration to mdBook

Generated documentation is automatically integrated into the project's mdBook documentation system:

1. **Symbolic Link Creation**: Create corresponding directory structure under `docs/bb-note/src/`
2. **SUMMARY.md Update**: Automatically add new documentation to the table of contents
3. **Structure Validation**: Ensure code directories and documentation directories correspond one-to-one

### Directory Mapping Example

```
Code Directory             →  Documentation Directory
arch/src/main/scala/       →  docs/bb-note/src/arch/src/main/scala/
bb-tests/workloads/        →  docs/bb-note/src/bb-tests/workloads/
scripts/docker/            →  docs/bb-note/src/scripts/docker/
```

## Common Issues and Troubleshooting

### Q1: Documentation generation fails with "path does not exist" error

**Cause**: The provided `target_path` does not exist or is inaccessible

**Solution**:
```bash
# Check if path exists
ls -la arch/src/main/scala/framework

# Ensure using relative path, do not start with /
# Correct: "arch/src/main/scala/framework"
# Wrong: "/arch/src/main/scala/framework"
```

### Q2: Generated documentation is of poor quality or inaccurate content

**Causes**:
- Few code files in directory or insufficient comments
- Wrong generation mode selected
- LLM API response anomaly

**Solutions**:
```bash
# 1. Check directory contents
find arch/src/main/scala/framework -name "*.scala" | head -10

# 2. Try update mode instead of create mode
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "update"
  }'

# 3. Check system logs
tail -f logs/doc-agent.log
```

### Q3: SUMMARY.md update fails

**Causes**:
- SUMMARY.md file permission issues
- File format does not meet expectations
- Concurrent update conflicts

**Solutions**:
```bash
# Check file permissions
ls -la docs/bb-note/src/SUMMARY.md

# Backup and reset SUMMARY.md
cp docs/bb-note/src/SUMMARY.md docs/bb-note/src/SUMMARY.md.backup

# Check file format
head -20 docs/bb-note/src/SUMMARY.md
```

### Q4: Symbolic link creation fails

**Causes**:
- Insufficient permissions for target directory
- Insufficient disk space
- File system does not support symbolic links

**Solutions**:
```bash
# Check disk space
df -h docs/

# Check permissions
ls -la docs/bb-note/src/

# Manually test symbolic link creation
ln -s ../../../arch/src/main/scala/framework docs/bb-note/src/arch/src/main/scala/framework
```

### Q5: API request timeout

**Causes**:
- Slow LLM API response
- Too many files in directory, long analysis time
- Network connection issues

**Solutions**:
```bash
# Increase request timeout
curl -X POST http://localhost:8080/doc/generate \
  --max-time 300 \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "create"
  }'

# Process large directories in batches
# Do not process arch/src/main/scala directly, but process its subdirectories
```

## Performance Optimization Recommendations

### 1. Batch Processing Optimization
```bash
# Use parallel processing (use cautiously, avoid API limits)
parallel -j 2 curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{\"target_path\": \"{}\", \"mode\": \"create\"}' \
  ::: arch/src/main/scala/framework arch/src/main/scala/builtin
```

### 2. Incremental Update Strategy
```bash
# Only update recently modified directories
find arch/src/main/scala -type d -mtime -7 | while read dir; do
  if [[ -f "$dir/README.md" ]]; then
    curl -X POST http://localhost:8080/doc/generate \
      -H "Content-Type: application/json" \
      -d "{\"target_path\": \"$dir\", \"mode\": \"update\"}"
  fi
done
```

### 3. Monitoring and Logging
```bash
# Monitor API response time
time curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "create"
  }'

# View detailed logs
tail -f logs/doc-agent.log | grep -E "(ERROR|WARN|Generation complete)"
```

## Configuration Guide

### Environment Variables

| Variable Name | Description | Default Value |
|--------|------|--------|
| `DOC_AGENT_PORT` | API service port | 8080 |
| `LLM_API_KEY` | LLM API key | Required to set |
| `LLM_API_URL` | LLM API endpoint | Required to set |
| `DOC_OUTPUT_BASE` | Documentation output base path | `docs/bb-note/src` |
| `TEMPLATE_BASE_PATH` | Template file base path | `workflow/prompts/doc` |

### Configuration File Example

Create `.env` file:
```bash
# LLM API configuration
LLM_API_KEY=your_api_key_here
LLM_API_URL=https://api.openai.com/v1/chat/completions

# Documentation system configuration
DOC_OUTPUT_BASE=docs/bb-note/src
TEMPLATE_BASE_PATH=workflow/prompts/doc

# Performance configuration
MAX_CONCURRENT_REQUESTS=3
REQUEST_TIMEOUT=300
```

## Development and Debugging

### Local Development Environment Setup

```bash
# 1. Install dependencies
cd workflow
npm install

# 2. Start Motia service
npm run dev

# 3. Test API connection
curl http://localhost:8080/health
```

### Debug Mode

```bash
# Enable verbose logging
export DEBUG=doc-agent:*
npm run dev

# Test individual components
node -e "
const { loadTemplate } = require('./steps/doc-agent/template_loader');
console.log(loadTemplate('rtl', 'arch/src/main/scala/test'));
"
```

## Version Information

- **Current Version**: 1.0.0
- **Motia Framework Version**: Compatible with v2.x
- **Supported Node.js Version**: >= 16.0.0
- **Last Updated**: December 2024

## Support and Feedback

If you encounter issues or need feature improvements, please:

1. Check the troubleshooting section of this document
2. Review system log files
3. Create an Issue in the project repository
4. Contact the development team

---

*This document is continuously maintained with system updates, please check regularly for the latest version.*
