#include "socket.h"
#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

SocketClient::SocketClient()
    : sock_fd(-1), socket_initialized(false), p(nullptr) {}

SocketClient::~SocketClient() { close(); }

bool SocketClient::init() {
  if (socket_initialized) {
    return true;
  }

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    fprintf(stderr, "Socket: Failed to create socket\n");
    return false;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(SOCKET_PORT);

  if (inet_pton(AF_INET, SOCKET_HOST, &server_addr.sin_addr) <= 0) {
    fprintf(stderr, "Socket: Invalid address/Address not supported\n");
    ::close(sock_fd);
    sock_fd = -1;
    return false;
  }

  if (connect(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) <
      0) {
    fprintf(stderr, "Socket: Connection failed to %s:%d\n", SOCKET_HOST,
            SOCKET_PORT);
    ::close(sock_fd);
    sock_fd = -1;
    return false;
  }

  socket_initialized = true;
  printf("Socket: Connected to %s:%d\n", SOCKET_HOST, SOCKET_PORT);
  return true;
}

void SocketClient::close() {
  if (sock_fd >= 0) {
    ::close(sock_fd);
    sock_fd = -1;
  }
  socket_initialized = false;
}

// Receive message header (peek first to get type)
bool SocketClient::recv_header(msg_header_t &header) {
  if (sock_fd < 0) {
    fprintf(stderr, "Socket: Not connected\n");
    return false;
  }

  ssize_t received = recv(sock_fd, &header, sizeof(header), MSG_PEEK);

  if (received < 0) {
    fprintf(stderr, "Socket: Failed to peek header\n");
    close();
    return false;
  } else if (received == 0) {
    fprintf(stderr, "Socket: Connection closed by remote\n");
    close();
    return false;
  }

  return true;
}

uint64_t SocketClient::send_and_wait(uint32_t funct, uint64_t xs1,
                                     uint64_t xs2) {
  // Auto-connect if not connected
  if (!socket_initialized) {
    if (!init()) {
      return 0;
    }
  }

  // Prepare and send CMD request
  cmd_req_t cmd_req;
  cmd_req.header.msg_type = MSG_TYPE_CMD_REQ;
  cmd_req.header.reserved = 0;
  cmd_req.funct = funct;
  cmd_req.padding = 0;
  cmd_req.xs1 = xs1;
  cmd_req.xs2 = xs2;

  if (!send_cmd_request(cmd_req)) {
    return 0;
  }

  // Loop to handle responses (CMD response or DMA requests)
  while (true) {
    // Peek message header to determine type
    msg_header_t header;
    if (!recv_header(header)) {
      return 0;
    }

    // Handle based on message type
    if (header.msg_type == MSG_TYPE_CMD_RESP) {
      // Receive CMD response
      cmd_resp_t cmd_resp;
      if (!recv_cmd_response(cmd_resp)) {
        return 0;
      }
      return cmd_resp.result;

    } else if (header.msg_type == MSG_TYPE_DMA_READ_REQ) {
      // Receive DMA read request
      dma_read_req_t dma_read_req;
      if (!recv_dma_read_request(dma_read_req)) {
        return 0;
      }

      // Handle DMA read
      uint64_t read_data =
          handle_dma_read(dma_read_req.addr, dma_read_req.size);

      // Send DMA read response
      dma_read_resp_t dma_read_resp;
      dma_read_resp.header.msg_type = MSG_TYPE_DMA_READ_RESP;
      dma_read_resp.header.reserved = 0;
      dma_read_resp.data = read_data;

      if (!send_dma_read_response(dma_read_resp)) {
        return 0;
      }

    } else if (header.msg_type == MSG_TYPE_DMA_WRITE_REQ) {
      // Receive DMA write request
      dma_write_req_t dma_write_req;
      if (!recv_dma_write_request(dma_write_req)) {
        return 0;
      }

      // Handle DMA write
      handle_dma_write(dma_write_req.addr, dma_write_req.data,
                       dma_write_req.size);

      // Send DMA write response
      dma_write_resp_t dma_write_resp;
      dma_write_resp.header.msg_type = MSG_TYPE_DMA_WRITE_RESP;
      dma_write_resp.header.reserved = 0;
      dma_write_resp.reserved = 0;

      if (!send_dma_write_response(dma_write_resp)) {
        return 0;
      }

    } else {
      fprintf(stderr, "Socket: Unknown message type %d\n", header.msg_type);
      close();
      return 0;
    }
  }
}
