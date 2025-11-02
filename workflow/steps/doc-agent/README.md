# Doc-Agent Workflow

Documentation generation workflow in the BuckyBall framework, providing automated code documentation generation functionality.

## API Usage Guide

### `generate`
**Endpoint**: `POST /doc/generate`

**Function**: Generate documentation for specified directory

**Parameters**:
- **`target_path`** [Required] - Target directory path
- **`mode`** [Required] - Generation mode, options: `"create"`, `"update"`

**Example**:
```bash
# Create new documentation for specified directory
bbdev doc --generate "--target_path arch/src/main/scala/framework --mode create"

# Update existing documentation
bbdev doc --generate "--target_path arch/src/main/scala/framework --mode update"
```

**Response**:
```json
{
  "traceId": "unique-trace-id",
  "status": "success",
  "message": "Documentation generated successfully"
}
```

## Supported Document Types

- RTL hardware documentation
- Test documentation
- Script documentation
- Simulator documentation
- Workflow documentation

## Important Notes

- Requires AI model API key configuration
- Generated documentation is automatically integrated into the mdBook system
- Supports symbolic link management and automatic SUMMARY.md updates
