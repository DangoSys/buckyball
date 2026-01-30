#ifndef MONITOR_COSIM_H_
#define MONITOR_COSIM_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>

// Forward declaration for VToyBuckyball
class VToyBuckyball;

// Socket protocol structures (match bebop/host/ipc/include/ipc/socket.h)
// Verilator uses different ports (7000-7002) to avoid conflict with Bebop's
// server (6000-6002)
#define SOCKET_CMD_PORT 7000
#define SOCKET_DMA_READ_PORT 7001
#define SOCKET_DMA_WRITE_PORT 7002

enum socket_msg_type_t : uint32_t {
  MSG_TYPE_CMD_REQ = 0,
  MSG_TYPE_CMD_RESP = 1,
  MSG_TYPE_DMA_READ_REQ = 2,
  MSG_TYPE_DMA_READ_RESP = 3,
  MSG_TYPE_DMA_WRITE_REQ = 4,
  MSG_TYPE_DMA_WRITE_RESP = 5
};

struct msg_header_t {
  uint32_t msg_type;
  uint32_t reserved;
};

struct cmd_req_t {
  msg_header_t header;
  uint32_t funct;
  uint32_t padding;
  uint64_t xs1;
  uint64_t xs2;
};

struct cmd_resp_t {
  msg_header_t header;
  uint64_t result;
};

struct dma_read_req_t {
  msg_header_t header;
  uint32_t size;
  uint32_t padding;
  uint64_t addr;
};

struct dma_read_resp_t {
  msg_header_t header;
  uint64_t data_lo;
  uint64_t data_hi;
};

struct dma_write_req_t {
  msg_header_t header;
  uint32_t size;
  uint32_t padding;
  uint64_t addr;
  uint64_t data_lo;
  uint64_t data_hi;
};

struct dma_write_resp_t {
  msg_header_t header;
  uint64_t reserved;
};

// TileLink transaction for pending DMA requests
struct tilelink_transaction_t {
  uint8_t opcode;   // TileLink opcode (0/1=Put, 4=Get)
  uint8_t source;   // Transaction ID
  uint32_t address; // Memory address
  uint8_t size;     // Transfer size (2^size bytes)
  uint16_t mask;    // Byte mask
  uint64_t data_lo; // Data low 64 bits (for write)
  uint64_t data_hi; // Data high 64 bits (for write)
};

// Cosim socket server for VToyBuckyball
class CosimServer {
public:
  CosimServer(VToyBuckyball *dut);
  ~CosimServer();

  // Initialize socket servers and start listening
  bool init();

  // Shutdown all connections
  void shutdown();

  // Main update function - called each cycle to handle socket I/O and drive DUT
  void update();

  // Check if server is running
  bool is_running() const { return server_running; }

private:
  VToyBuckyball *dut;
  std::atomic<bool> server_running;

  // Socket file descriptors
  int cmd_server_fd;
  int dma_read_server_fd;
  int dma_write_server_fd;
  int cmd_client_fd;
  int dma_read_client_fd;
  int dma_write_client_fd;

  // Background threads for socket I/O
  std::thread cmd_thread;
  std::thread dma_read_thread;
  std::thread dma_write_thread;

  // CMD queue (Bebop -> Verilator)
  std::queue<cmd_req_t> cmd_queue;
  std::mutex cmd_mutex;
  std::condition_variable cmd_cv;

  // TileLink request queue (Verilator -> Bebop)
  std::queue<tilelink_transaction_t> tl_req_queue;
  std::mutex tl_req_mutex;

  // TileLink response queue (Bebop -> Verilator)
  std::queue<tilelink_transaction_t> tl_resp_queue;
  std::mutex tl_resp_mutex;

  // Current CMD being processed
  bool cmd_in_progress;
  cmd_req_t current_cmd;

  // Socket thread functions
  void cmd_server_thread();
  void dma_read_server_thread();
  void dma_write_server_thread();

  // Socket helpers
  int create_server_socket(int port);
  int accept_client(int server_fd);
  bool recv_all(int fd, void *buf, size_t len);
  bool send_all(int fd, const void *buf, size_t len);

  // CMD processing
  void handle_cmd_request();
  void drive_cmd_interface();

  // TileLink/DMA processing
  void sample_tilelink_request();
  void drive_tilelink_response();
  void process_dma_read_request(const tilelink_transaction_t &tx);
  void process_dma_write_request(const tilelink_transaction_t &tx);
};

#endif // MONITOR_COSIM_H_
