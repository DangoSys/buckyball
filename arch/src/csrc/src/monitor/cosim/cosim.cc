#ifdef COSIM

#include "monitor/cosim.h"
#include "VToyBuckyball.h"
#include "utils/debug.h"
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

CosimServer::CosimServer(VToyBuckyball *dut)
    : dut(dut), server_running(false), cmd_server_fd(-1),
      dma_read_server_fd(-1), dma_write_server_fd(-1), cmd_client_fd(-1),
      dma_read_client_fd(-1), dma_write_client_fd(-1), cmd_in_progress(false) {}

CosimServer::~CosimServer() { shutdown(); }

bool CosimServer::init() {
  Log("Initializing COSIM socket server...");

  // Create server sockets
  cmd_server_fd = create_server_socket(SOCKET_CMD_PORT);
  dma_read_server_fd = create_server_socket(SOCKET_DMA_READ_PORT);
  dma_write_server_fd = create_server_socket(SOCKET_DMA_WRITE_PORT);

  if (cmd_server_fd < 0 || dma_read_server_fd < 0 || dma_write_server_fd < 0) {
    panic("Failed to create server sockets");
    return false;
  }

  Log("Socket servers listening on ports %d, %d, %d", SOCKET_CMD_PORT,
      SOCKET_DMA_READ_PORT, SOCKET_DMA_WRITE_PORT);

  server_running = true;

  // Start background threads
  cmd_thread = std::thread(&CosimServer::cmd_server_thread, this);
  dma_read_thread = std::thread(&CosimServer::dma_read_server_thread, this);
  dma_write_thread = std::thread(&CosimServer::dma_write_server_thread, this);

  Log("COSIM server initialized successfully");
  return true;
}

void CosimServer::shutdown() {
  if (!server_running)
    return;

  Log("Shutting down COSIM server...");
  server_running = false;

  // Wake up all threads
  cmd_cv.notify_all();

  // Close client connections
  if (cmd_client_fd >= 0)
    close(cmd_client_fd);
  if (dma_read_client_fd >= 0)
    close(dma_read_client_fd);
  if (dma_write_client_fd >= 0)
    close(dma_write_client_fd);

  // Close server sockets
  if (cmd_server_fd >= 0)
    close(cmd_server_fd);
  if (dma_read_server_fd >= 0)
    close(dma_read_server_fd);
  if (dma_write_server_fd >= 0)
    close(dma_write_server_fd);

  // Join threads
  if (cmd_thread.joinable())
    cmd_thread.join();
  if (dma_read_thread.joinable())
    dma_read_thread.join();
  if (dma_write_thread.joinable())
    dma_write_thread.join();

  Log("COSIM server shutdown complete");
}

int CosimServer::create_server_socket(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    panic("socket() failed: %s", strerror(errno));
    return -1;
  }

  // Set SO_REUSEADDR
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    panic("setsockopt() failed: %s", strerror(errno));
    close(fd);
    return -1;
  }

  // Bind
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    panic("bind() failed on port %d: %s", port, strerror(errno));
    close(fd);
    return -1;
  }

  // Listen
  if (listen(fd, 1) < 0) {
    panic("listen() failed: %s", strerror(errno));
    close(fd);
    return -1;
  }

  return fd;
}

int CosimServer::accept_client(int server_fd) {
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);
  int client_fd =
      accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
  if (client_fd < 0) {
    panic("accept() failed: %s", strerror(errno));
    return -1;
  }
  Log("Accepted client connection from %s:%d", inet_ntoa(client_addr.sin_addr),
      ntohs(client_addr.sin_port));

  // Set TCP_NODELAY to disable Nagle's algorithm
  int opt = 1;
  if (setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
    Log("Failed to set TCP_NODELAY: %s", strerror(errno));
  }

  return client_fd;
}

bool CosimServer::recv_all(int fd, void *buf, size_t len) {
  size_t received = 0;
  while (received < len) {
    ssize_t n = recv(fd, (char *)buf + received, len - received, 0);
    if (n <= 0) {
      if (n < 0)
        panic("recv() failed: %s", strerror(errno));
      return false;
    }
    received += n;
  }
  return true;
}

bool CosimServer::send_all(int fd, const void *buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = send(fd, (const char *)buf + sent, len - sent, 0);
    if (n <= 0) {
      if (n < 0)
        panic("send() failed: %s", strerror(errno));
      return false;
    }
    sent += n;
  }
  return true;
}

// ============ CMD Server Thread ============
void CosimServer::cmd_server_thread() {
  Log("CMD server thread started");

  // Accept client connection
  cmd_client_fd = accept_client(cmd_server_fd);
  if (cmd_client_fd < 0)
    return;

  // Process CMD requests
  while (server_running) {
    cmd_req_t req;
    if (!recv_all(cmd_client_fd, &req, sizeof(req))) {
      Log("CMD client disconnected");
      break;
    }

    if (req.header.msg_type != MSG_TYPE_CMD_REQ) {
      panic("Invalid CMD message type: %u", req.header.msg_type);
      continue;
    }

    // Enqueue CMD request
    {
      std::lock_guard<std::mutex> lock(cmd_mutex);
      cmd_queue.push(req);
    }
    cmd_cv.notify_one();

    // Wait for CMD completion (io_busy goes low)
    // The response will be sent from the main update loop
  }

  Log("CMD server thread exiting");
}

// ============ DMA Read Server Thread ============
void CosimServer::dma_read_server_thread() {
  Log("DMA Read server thread started");

  // Accept client connection
  dma_read_client_fd = accept_client(dma_read_server_fd);
  if (dma_read_client_fd < 0)
    return;
  Log("DMA Read client accepted");

  // Process DMA read responses from client
  while (server_running) {
    Log("DMA Read thread: waiting for response...");
    dma_read_resp_t resp;
    // Use non-blocking recv to avoid deadlock
    int flags = fcntl(dma_read_client_fd, F_GETFL, 0);
    fcntl(dma_read_client_fd, F_SETFL, flags | O_NONBLOCK);

    ssize_t n = recv(dma_read_client_fd, &resp, sizeof(resp), 0);

    fcntl(dma_read_client_fd, F_SETFL, flags);

    if (n > 0) {
      Log("DMA Read thread: received msg_type=%u, size=%ld",
          resp.header.msg_type, n);

      if (resp.header.msg_type == MSG_TYPE_DMA_READ_RESP) {
        // Enqueue TileLink response
        tilelink_transaction_t tx;
        tx.opcode = 1; // AccessAckData
        tx.data_lo = resp.data_lo;
        tx.data_hi = resp.data_hi;

        {
          std::lock_guard<std::mutex> lock(tl_resp_mutex);
          tl_resp_queue.push(tx);
        }
      } else {
        panic("Invalid DMA Read response type: %u", resp.header.msg_type);
      }
    } else if (n == 0) {
      Log("DMA Read client disconnected");
      break;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      Log("DMA Read recv error: %s", strerror(errno));
      break;
    }

    // Sleep briefly to avoid busy-waiting
    usleep(100);
  }

  Log("DMA Read server thread exiting");
}

// ============ DMA Write Server Thread ============
void CosimServer::dma_write_server_thread() {
  Log("DMA Write server thread started");

  // Accept client connection
  dma_write_client_fd = accept_client(dma_write_server_fd);
  if (dma_write_client_fd < 0)
    return;

  // Process DMA write responses from client
  while (server_running) {
    dma_write_resp_t resp;
    // Use non-blocking recv to avoid deadlock
    int flags = fcntl(dma_write_client_fd, F_GETFL, 0);
    fcntl(dma_write_client_fd, F_SETFL, flags | O_NONBLOCK);

    ssize_t n = recv(dma_write_client_fd, &resp, sizeof(resp), 0);

    fcntl(dma_write_client_fd, F_SETFL, flags);

    if (n > 0) {
      Log("DMA Write thread: received msg_type=%u", resp.header.msg_type);

      if (resp.header.msg_type == MSG_TYPE_DMA_WRITE_RESP) {
        // Enqueue TileLink response
        tilelink_transaction_t tx;
        tx.opcode = 0; // AccessAck

        {
          std::lock_guard<std::mutex> lock(tl_resp_mutex);
          tl_resp_queue.push(tx);
        }
      } else {
        panic("Invalid DMA Write response type: %u", resp.header.msg_type);
      }
    } else if (n == 0) {
      Log("DMA Write client disconnected");
      break;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      Log("DMA Write recv error: %s", strerror(errno));
      break;
    }

    // Sleep briefly to avoid busy-waiting
    usleep(100);
  }

  Log("DMA Write server thread exiting");
}

// ============ Main Update (called each cycle) ============
void CosimServer::update() {
  // 1. Drive CMD interface (io_cmd_*)
  drive_cmd_interface();

  // 2. Sample TileLink requests (auto_widget_anon_out_a_*)
  sample_tilelink_request();

  // 3. Drive TileLink responses (auto_widget_anon_out_d_*)
  drive_tilelink_response();
}

void CosimServer::drive_cmd_interface() {
  // Check if there's a pending CMD request
  if (!cmd_in_progress) {
    std::lock_guard<std::mutex> lock(cmd_mutex);
    if (!cmd_queue.empty()) {
      current_cmd = cmd_queue.front();
      cmd_queue.pop();
      cmd_in_progress = true;

      // Drive io_cmd signals
      dut->io_cmd_valid = 1;
      dut->io_cmd_bits_inst_funct = current_cmd.funct & 0x7F;
      dut->io_cmd_bits_rs1 = current_cmd.xs1;
      dut->io_cmd_bits_rs2 = current_cmd.xs2;

      Log("CMD Request: funct=%u, xs1=0x%lx, xs2=0x%lx", current_cmd.funct,
          current_cmd.xs1, current_cmd.xs2);
    } else {
      // No pending CMD, clear io_cmd_valid
      dut->io_cmd_valid = 0;
    }
  } else {
    // CMD in progress, check if accelerator is ready
    if (dut->io_cmd_ready == 1) {
      // CMD accepted, clear valid
      dut->io_cmd_valid = 0;
      cmd_in_progress = false;

      // Send response immediately
      cmd_resp_t resp;
      resp.header.msg_type = MSG_TYPE_CMD_RESP;
      resp.header.reserved = 0;
      resp.result = 1; // Success

      if (cmd_client_fd >= 0) {
        send_all(cmd_client_fd, &resp, sizeof(resp));
      }

      Log("CMD Response: result=0x%lx", resp.result);
    }
  }
}

void CosimServer::sample_tilelink_request() {
  // Check if TileLink Channel A has a valid request
  static int no_request_count = 0;

  if (dut->auto_widget_anon_out_a_valid == 1) {
    tilelink_transaction_t tx;
    tx.opcode = dut->auto_widget_anon_out_a_bits_opcode;
    tx.source = dut->auto_widget_anon_out_a_bits_source;
    tx.address = dut->auto_widget_anon_out_a_bits_address;
    tx.size = dut->auto_widget_anon_out_a_bits_size;
    tx.mask = dut->auto_widget_anon_out_a_bits_mask;

    // Read data from 128-bit VlWide array (4 x 32-bit words)
    // VlWide<4> means 4 words of 32 bits each = 128 bits total
    uint32_t *data_words = dut->auto_widget_anon_out_a_bits_data;
    tx.data_lo = ((uint64_t)data_words[1] << 32) | (uint64_t)data_words[0];
    tx.data_hi = ((uint64_t)data_words[3] << 32) | (uint64_t)data_words[2];

    Log("TileLink Request: opcode=%u, addr=0x%x, size=%u, source=%u, ready=%d",
        tx.opcode, tx.address, tx.size, tx.source,
        dut->auto_widget_anon_out_a_ready);

    no_request_count = 0;

    // Process based on opcode
    if (tx.opcode == 4) {
      // Get (Read)
      process_dma_read_request(tx);
    } else if (tx.opcode == 0 || tx.opcode == 1) {
      // PutFullData or PutPartialData (Write)
      process_dma_write_request(tx);
    }

    // Set ready after handling
    dut->auto_widget_anon_out_a_ready = 1;
  } else {
    // No request
    if (++no_request_count % 100 == 0) {
      Log("No TileLink requests yet (valid=%d, ready=%d)",
          dut->auto_widget_anon_out_a_valid, dut->auto_widget_anon_out_a_ready);
    }
    dut->auto_widget_anon_out_a_ready = 1;
  }
}

void CosimServer::drive_tilelink_response() {
  // Check if there's a pending TileLink response
  std::lock_guard<std::mutex> lock(tl_resp_mutex);
  if (!tl_resp_queue.empty() && dut->auto_widget_anon_out_d_ready == 1) {
    tilelink_transaction_t tx = tl_resp_queue.front();
    tl_resp_queue.pop();

    // Drive Channel D signals
    dut->auto_widget_anon_out_d_valid = 1;
    dut->auto_widget_anon_out_d_bits_opcode = tx.opcode;
    dut->auto_widget_anon_out_d_bits_source = tx.source;

    if (tx.opcode == 1) {
      // AccessAckData (read response)
      // Write data to 128-bit VlWide array (4 x 32-bit words)
      uint32_t *data_words = dut->auto_widget_anon_out_d_bits_data;
      data_words[0] = (uint32_t)(tx.data_lo & 0xFFFFFFFF);
      data_words[1] = (uint32_t)(tx.data_lo >> 32);
      data_words[2] = (uint32_t)(tx.data_hi & 0xFFFFFFFF);
      data_words[3] = (uint32_t)(tx.data_hi >> 32);
      Log("TileLink Response: opcode=%u (AckData), data=0x%lx%lx", tx.opcode,
          tx.data_hi, tx.data_lo);
    } else {
      // AccessAck (write response)
      Log("TileLink Response: opcode=%u (Ack)", tx.opcode);
    }
  } else {
    // No pending response
    dut->auto_widget_anon_out_d_valid = 0;
  }
}

void CosimServer::process_dma_read_request(const tilelink_transaction_t &tx) {
  // Send DMA read request to client
  dma_read_req_t req;
  req.header.msg_type = MSG_TYPE_DMA_READ_REQ;
  req.header.reserved = 0;
  req.addr = tx.address;
  req.size = 1 << tx.size; // Convert 2^size to bytes
  req.padding = 0;

  Log("process_dma_read_request: about to send request, fd=%d",
      dma_read_client_fd);
  if (dma_read_client_fd >= 0) {
    Log("process_dma_read_request: sending request...");
    // Disable Nagle's algorithm to ensure immediate transmission
    int opt = 1;
    setsockopt(dma_read_client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    send_all(dma_read_client_fd, &req, sizeof(req));
    Log("process_dma_read_request: request sent: addr=0x%lx, size=%u", req.addr,
        req.size);
  } else {
    Log("process_dma_read_request: dma_read_client_fd is invalid!");
  }

  // Response will be handled in dma_read_server_thread
}

void CosimServer::process_dma_write_request(const tilelink_transaction_t &tx) {
  // Send DMA write request to client
  dma_write_req_t req;
  req.header.msg_type = MSG_TYPE_DMA_WRITE_REQ;
  req.header.reserved = 0;
  req.addr = tx.address;
  req.size = 1 << tx.size; // Convert 2^size to bytes
  req.padding = 0;
  req.data_lo = tx.data_lo;
  req.data_hi = tx.data_hi;

  if (dma_write_client_fd >= 0) {
    send_all(dma_write_client_fd, &req, sizeof(req));
    Log("DMA Write Request: addr=0x%lx, size=%u, data=0x%lx%lx", req.addr,
        req.size, req.data_hi, req.data_lo);
  }

  // Response will be handled in dma_write_server_thread
}
#endif
