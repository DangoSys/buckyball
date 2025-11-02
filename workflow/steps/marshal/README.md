# Marshal Workflow

Marshal workflow in the BuckyBall framework, used to build and launch the Marshal component.

## API Usage Guide

### `build`
**Endpoint**: `POST /marshal/build`

**Function**: Build Marshal component

**Parameters**: No specific parameters

**Example**:
```bash
bbdev marshal --build
```

### `launch`
**Endpoint**: `POST /marshal/launch`

**Function**: Launch Marshal service

**Parameters**: No specific parameters

**Example**:
```bash
bbdev marshal --launch
```

## Typical Workflow

```bash
# 1. Build Marshal
bbdev marshal --build

# 2. Launch Marshal service
bbdev marshal --launch
```

## Response Format

All API calls return a unified format:
```json
{
  "status": 200,
  "body": {
    "success": true,
    "processing": false,
    "return_code": 0
  }
}
```
