#include "pegasus.h"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Open XDMA device files and mmap BAR0
PegasusXDMA *xdma_open(const char *dev_prefix) {
    auto *xdma = new PegasusXDMA();
    xdma->bar0_size = 0x1000;  // 4 KB BAR0 for SCU registers
    xdma->bar0_base = nullptr;

    std::string prefix(dev_prefix);

    // Open H2C (host-to-card, i.e., write to FPGA)
    std::string h2c_path = prefix + "_h2c_0";
    xdma->h2c_fd = open(h2c_path.c_str(), O_WRONLY);
    if (xdma->h2c_fd < 0) {
        fprintf(stderr, "[pegasus] Failed to open %s: %s\n",
                h2c_path.c_str(), strerror(errno));
        delete xdma;
        return nullptr;
    }

    // Open C2H (card-to-host, i.e., read from FPGA)
    std::string c2h_path = prefix + "_c2h_0";
    xdma->c2h_fd = open(c2h_path.c_str(), O_RDONLY);
    if (xdma->c2h_fd < 0) {
        fprintf(stderr, "[pegasus] Failed to open %s: %s\n",
                c2h_path.c_str(), strerror(errno));
        close(xdma->h2c_fd);
        delete xdma;
        return nullptr;
    }

    // Open UART C2H stream (channel 1)
    std::string uart_path = prefix + "_c2h_1";
    xdma->uart_fd = open(uart_path.c_str(), O_RDONLY | O_NONBLOCK);
    if (xdma->uart_fd < 0) {
        // Non-fatal: UART stream may not be configured
        fprintf(stderr, "[pegasus] Warning: Failed to open UART stream %s: %s\n",
                uart_path.c_str(), strerror(errno));
        xdma->uart_fd = -1;
    }

    // Open and mmap BAR0 (user BAR for MMIO)
    std::string bar_path = prefix + "_user";
    xdma->bar_fd = open(bar_path.c_str(), O_RDWR | O_SYNC);
    if (xdma->bar_fd < 0) {
        fprintf(stderr, "[pegasus] Failed to open BAR0 %s: %s\n",
                bar_path.c_str(), strerror(errno));
        close(xdma->h2c_fd);
        close(xdma->c2h_fd);
        if (xdma->uart_fd >= 0) close(xdma->uart_fd);
        delete xdma;
        return nullptr;
    }

    xdma->bar0_base = mmap(nullptr, xdma->bar0_size,
                           PROT_READ | PROT_WRITE, MAP_SHARED,
                           xdma->bar_fd, 0);
    if (xdma->bar0_base == MAP_FAILED) {
        fprintf(stderr, "[pegasus] Failed to mmap BAR0: %s\n", strerror(errno));
        close(xdma->h2c_fd);
        close(xdma->c2h_fd);
        if (xdma->uart_fd >= 0) close(xdma->uart_fd);
        close(xdma->bar_fd);
        delete xdma;
        return nullptr;
    }

    return xdma;
}

void xdma_close(PegasusXDMA *xdma) {
    if (!xdma) return;
    if (xdma->bar0_base && xdma->bar0_base != MAP_FAILED) {
        munmap(xdma->bar0_base, xdma->bar0_size);
    }
    if (xdma->bar_fd >= 0)  close(xdma->bar_fd);
    if (xdma->h2c_fd >= 0)  close(xdma->h2c_fd);
    if (xdma->c2h_fd >= 0)  close(xdma->c2h_fd);
    if (xdma->uart_fd >= 0) close(xdma->uart_fd);
    delete xdma;
}

// BAR0 MMIO (32-bit aligned reads/writes)
void mmio_write32(PegasusXDMA *xdma, uint64_t offset, uint32_t val) {
    assert(xdma && xdma->bar0_base);
    assert(offset + 4 <= xdma->bar0_size);
    volatile uint32_t *reg = reinterpret_cast<volatile uint32_t *>(
        static_cast<uint8_t *>(xdma->bar0_base) + offset);
    *reg = val;
}

uint32_t mmio_read32(PegasusXDMA *xdma, uint64_t offset) {
    assert(xdma && xdma->bar0_base);
    assert(offset + 4 <= xdma->bar0_size);
    volatile uint32_t *reg = reinterpret_cast<volatile uint32_t *>(
        static_cast<uint8_t *>(xdma->bar0_base) + offset);
    return *reg;
}

// DMA write: host → FPGA HBM2 via H2C channel
ssize_t dma_write(PegasusXDMA *xdma, uint64_t fpga_addr,
                  const void *buf, size_t len) {
    assert(xdma && xdma->h2c_fd >= 0);
    return pwrite(xdma->h2c_fd, buf, len, static_cast<off_t>(fpga_addr));
}

// DMA read: FPGA HBM2 → host via C2H channel
ssize_t dma_read(PegasusXDMA *xdma, uint64_t fpga_addr,
                 void *buf, size_t len) {
    assert(xdma && xdma->c2h_fd >= 0);
    return pread(xdma->c2h_fd, buf, len, static_cast<off_t>(fpga_addr));
}
