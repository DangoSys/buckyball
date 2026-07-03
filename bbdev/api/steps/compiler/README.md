# Compiler Workflow

Compiler build workflow in the Buckyball framework for building the Buckyball compiler toolchain.

## API Usage

### `build`
**Endpoint**: `POST /compiler/build`

**Function**: Build Buckyball compiler

**Parameters**:
- `chip`: compiler chip package used as `BUDDY_EXTERNAL_DIALECTS_DIR`; valid values include `toy`, `goban`, and `pebble`
- `stable`: optional boolean flag; for `pebble`, build the compiler package that can use the stable LLVM backend lowering path

**Example**:
```bash
bbdev compiler --build '--chip toy'
bbdev compiler --build '--chip pebble --stable'
```

**Response**:
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

## Notes

- Ensure the system has necessary build tools and dependencies
- `pebble` supports both custom `xbuckyball` lowering and stable LLVM backend lowering; the selected lowering mode is controlled by pass/workload options
