#include "pegasus.h"

#include <cstdio>
#include <unistd.h>

// Non-blocking UART poll: read bytes from XDMA C2H channel 1, write to stdout
void uart_poll(PegasusXDMA *xdma) {
    if (xdma->uart_fd < 0) return;

    static char buf[4096];
    while (true) {
        ssize_t n = read(xdma->uart_fd, buf, sizeof(buf));
        if (n <= 0) break;  // EAGAIN or no data (non-blocking fd)
        fwrite(buf, 1, static_cast<size_t>(n), stdout);
        fflush(stdout);
    }
}
