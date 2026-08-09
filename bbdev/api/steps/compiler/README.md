# Compiler Workflow

Compiler build workflow in the Buckyball framework for building the Buckyball compiler toolchain.

## API Usage

### `build`
**Endpoint**: `POST /compiler/build`

**Function**: Build Buckyball compiler

**Parameters**:
- `core`: direct compiler Core target, such as `toy`, `goban`, or `pebble`
- `chip`: Chip runtime target; its `chip.toml` selects the default compiler Core. Specify exactly one of `core` or `chip`.
- `stable`: optional boolean flag; for `pebble`, build the compiler package that can use the stable LLVM backend lowering path

**Example**:
```bash
bbdev compiler --build '--chip toy'
bbdev compiler --build '--core goban'
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
