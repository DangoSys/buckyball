#ifndef _SOCKET_H
#define _SOCKET_H

#include <cstdint>

// Socket configuration
#define SOCKET_PORT 9999
#define SOCKET_HOST "127.0.0.1"

// Message types for socket communication
enum socket_msg_type_t : uint32_t {
  MSG_TYPE_CMD_REQ = 0,       // Command request from client
  MSG_TYPE_CMD_RESP = 1,      // Command response from server
  MSG_TYPE_DMA_READ_REQ = 2,  // DMA read request from server
  MSG_TYPE_DMA_READ_RESP = 3, // DMA read response from client
  MSG_TYPE_DMA_WRITE_REQ = 4, // DMA write request from server
  MSG_TYPE_DMA_WRITE_RESP = 5 // DMA write response from client
};

// Common message header
struct msg_header_t {
  uint32_t msg_type; // socket_msg_type_t
  uint32_t reserved;
};

// Command request from client (CMD path)
struct cmd_req_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_CMD_REQ
  uint32_t funct;
  uint32_t padding;
  uint64_t xs1;
  uint64_t xs2;
};

// Command response from server (CMD path)
struct cmd_resp_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_CMD_RESP
  uint64_t result;
};

// DMA read request from server (DMA path)
struct dma_read_req_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_DMA_READ_REQ
  uint32_t size;       // Size in bytes (1, 2, 4, or 8)
  uint32_t padding;
  uint64_t addr; // Memory address
};

// DMA read response from client (DMA path)
struct dma_read_resp_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_DMA_READ_RESP
  uint64_t data;       // Read data
};

// DMA write request from server (DMA path)
struct dma_write_req_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_DMA_WRITE_REQ
  uint32_t size;       // Size in bytes (1, 2, 4, or 8)
  uint32_t padding;
  uint64_t addr; // Memory address
  uint64_t data; // Write data
};

// DMA write response from client (DMA path)
struct dma_write_resp_t {
  msg_header_t header; // header.msg_type = MSG_TYPE_DMA_WRITE_RESP
  uint64_t reserved;   // Reserved for future use
};

// Forward declaration
class processor_t;

// Socket client class
class SocketClient {
public:
  SocketClient();
  ~SocketClient();

  // Initialize and connect to socket server
  bool init();

  // Close socket connection
  void close();

  // Set processor for DMA operations
  void set_processor(processor_t *p) { this->p = p; }

  // Send request and wait for response (handles DMA requests during wait)
  uint64_t send_and_wait(uint32_t funct, uint64_t xs1, uint64_t xs2);

  // Check if socket is connected
  bool is_connected() const { return socket_initialized; }

private:
  int sock_fd;
  bool socket_initialized;
  processor_t *p;

  // CMD path functions
  bool send_cmd_request(const cmd_req_t &req);
  bool recv_cmd_response(cmd_resp_t &resp);

  // DMA path functions
  bool recv_dma_read_request(dma_read_req_t &req);
  bool send_dma_read_response(const dma_read_resp_t &resp);
  bool recv_dma_write_request(dma_write_req_t &req);
  bool send_dma_write_response(const dma_write_resp_t &resp);

  // Low-level recv/send
  bool recv_header(msg_header_t &header);

  // DMA handlers
  uint64_t handle_dma_read(uint64_t addr, uint32_t size);
  void handle_dma_write(uint64_t addr, uint64_t data, uint32_t size);
};

#endif // _SOCKET_H
