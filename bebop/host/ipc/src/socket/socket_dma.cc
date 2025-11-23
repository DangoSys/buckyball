#include "ipc/socket.h"
#include <cstdio>
#include <sys/socket.h>

// DMA path: receive DMA read request
bool SocketClient::recv_dma_read_request(dma_read_req_t &req) {
  if (sock_fd < 0) {
    fprintf(stderr, "Socket: Not connected, cannot receive DMA read request\n");
    return false;
  }

  ssize_t received = recv(sock_fd, &req, sizeof(req), 0);

  if (received < 0) {
    fprintf(stderr, "Socket: Failed to receive DMA read request\n");
    close();
    return false;
  } else if (received == 0) {
    fprintf(stderr, "Socket: Connection closed by remote\n");
    close();
    return false;
  }

  return true;
}

// DMA path: send DMA read response
bool SocketClient::send_dma_read_response(const dma_read_resp_t &resp) {
  if (sock_fd < 0) {
    fprintf(stderr, "Socket: Not connected, cannot send DMA read response\n");
    return false;
  }

  ssize_t sent = send(sock_fd, &resp, sizeof(resp), 0);
  if (sent < 0) {
    fprintf(stderr, "Socket: Failed to send DMA read response\n");
    close();
    return false;
  }

  return true;
}

// DMA path: receive DMA write request
bool SocketClient::recv_dma_write_request(dma_write_req_t &req) {
  if (sock_fd < 0) {
    fprintf(stderr,
            "Socket: Not connected, cannot receive DMA write request\n");
    return false;
  }

  ssize_t received = recv(sock_fd, &req, sizeof(req), 0);

  if (received < 0) {
    fprintf(stderr, "Socket: Failed to receive DMA write request\n");
    close();
    return false;
  } else if (received == 0) {
    fprintf(stderr, "Socket: Connection closed by remote\n");
    close();
    return false;
  }

  return true;
}

// DMA path: send DMA write response
bool SocketClient::send_dma_write_response(const dma_write_resp_t &resp) {
  if (sock_fd < 0) {
    fprintf(stderr, "Socket: Not connected, cannot send DMA write response\n");
    return false;
  }

  ssize_t sent = send(sock_fd, &resp, sizeof(resp), 0);
  if (sent < 0) {
    fprintf(stderr, "Socket: Failed to send DMA write response\n");
    close();
    return false;
  }

  return true;
}

// DMA handlers
uint64_t SocketClient::handle_dma_read(uint64_t addr, uint32_t size) {
  if (!dma_read_cb) {
    fprintf(stderr, "Socket: DMA read callback not set\n");
    return 0;
  }
  uint64_t value = dma_read_cb(addr, size);
  printf("Socket: DMA read addr=0x%lx size=%d value=0x%lx\n", addr, size,
         value);
  return value;
}

void SocketClient::handle_dma_write(uint64_t addr, uint64_t data,
                                    uint32_t size) {
  if (!dma_write_cb) {
    fprintf(stderr, "Socket: DMA write callback not set\n");
    return;
  }
  dma_write_cb(addr, data, size);
  printf("Socket: DMA write addr=0x%lx size=%d data=0x%lx\n", addr, size, data);
}
