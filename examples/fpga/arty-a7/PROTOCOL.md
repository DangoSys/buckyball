# Serial loader and MMIO

The 115200 8N1 link uses stop-and-wait request/response packets while the CPU is
held in reset. After RUN it becomes the software-controlled UART.

Request: `A5 5A op:u8 seq:u8 address:u32le length:u16le [WRITE data] crc:u16le`.
Response: `5A A5 (op|80):u8 seq:u8 status:u8 length:u16le [data] crc:u16le`.
CRC-16/CCITT-FALSE uses initial FFFF and polynomial 1021, covering op through
payload, excluding sync bytes and CRC. It detects corruption, not authentication.

| Operation | Behavior |
|---|---|
| 1 WRITE | 1..256 bytes; RAM changes only after frame CRC validation |
| 2 READ | 1..256 bytes |
| 3 RUN | Address 80000000, length 0; requires a previous successful WRITE |
| 4 INFO | Address/length 0; returns `PA7 01`, RAM size u32le, clock Hz u32le |

Status: 0 success, 1 CRC error, 2 invalid range, 3 unsupported format/operation,
5 RUN before any write. Oversized or timed-out frames are discarded. Reset and
retry after a host timeout. The CPU is released only after the RUN reply drains.
Hardware checks individual frames; `host/load.py` verifies the entire uploaded
image by readback before RUN. Reset can interrupt an upload; restart the upload.

AXI UART base: `0x60000000`.

| Offset | Access | Meaning |
|---|---|---|
| 00 | write u8 | TX byte; AXI write data stalls while busy |
| 04 | read/write u32 | bit0 TX ready, bit1 RX available, bit2 overflow, bit3 frame error; write-one-clear bits2/3 |
| 08 | read u32 | bit31 valid, low byte RX data; pop on read response handshake |
| 10 | write u32/read u64 | Exit code; zero means success; read bit32 indicates a software write |
| 18 | read u64/byte writes | Scratch register |
| 20 | read u64 | Core-clock cycle count since CPU reset |

The peripheral supports single-beat naturally aligned accesses; unsupported
addresses/bursts return SLVERR. The RAM supports naturally aligned FIXED/INCR
bursts, 128-bit data, byte write enables, independent read/write IDs and
backpressure. WRAP and exclusive accesses are not supported by this platform.
