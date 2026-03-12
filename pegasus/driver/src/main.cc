#include "pegasus.h"

#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

// ============================================================
// Usage:
//   pegasus-run <elf>               Bare-metal: poll tohost, exit on done
//   pegasus-run --linux <elf>       Linux mode: forward UART, Ctrl+C to stop
// ============================================================

static volatile bool g_interrupted = false;

static void sigint_handler(int) {
    g_interrupted = true;
}

static void print_usage(const char *argv0) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <elf>            bare-metal mode (poll tohost)\n", argv0);
    fprintf(stderr, "  %s --linux <elf>    Linux mode (UART only, Ctrl+C to exit)\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    bool linux_mode = false;
    const char *elf_path = nullptr;

    if (argc >= 3 && strcmp(argv[1], "--linux") == 0) {
        linux_mode = true;
        elf_path   = argv[2];
    } else if (argc >= 2 && argv[1][0] != '-') {
        elf_path = argv[1];
    } else {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "[pegasus] Mode: %s\n", linux_mode ? "Linux" : "bare-metal");
    fprintf(stderr, "[pegasus] ELF: %s\n", elf_path);

    // 1. Parse ELF
    ELFInfo elf = elf_parse(elf_path);

    // 2. Open XDMA device
    PegasusXDMA *xdma = xdma_open("/dev/xdma0");
    if (!xdma) {
        fprintf(stderr, "[pegasus] Failed to open XDMA device\n");
        return 1;
    }

    // 3. Reset DUT
    fprintf(stderr, "[pegasus] Resetting DUT...\n");
    scu_reset(xdma);

    // 4. Load ELF into HBM2
    fprintf(stderr, "[pegasus] Loading ELF into HBM2...\n");
    elf_load(xdma, &elf);

    // 5. Start DUT (free-running)
    fprintf(stderr, "[pegasus] Starting DUT...\n");
    scu_run(xdma);

    // 6. Run loop
    signal(SIGINT, sigint_handler);

    if (linux_mode) {
        // Linux mode: forward UART output until Ctrl+C
        fprintf(stderr, "[pegasus] Linux mode running. Press Ctrl+C to stop.\n");
        while (!g_interrupted) {
            uart_poll(xdma);
            usleep(1000);  // 1ms poll interval
        }
        fprintf(stderr, "\n[pegasus] Interrupted by user.\n");

    } else {
        // Bare-metal mode: poll tohost + forward UART
        if (elf.tohost_paddr == 0) {
            fprintf(stderr, "[pegasus] Warning: tohost not found in ELF; "
                            "cannot detect program exit. Running indefinitely.\n");
            while (!g_interrupted) {
                uart_poll(xdma);
                usleep(1000);
            }
        } else {
            uint64_t tohost_hbm2 = SOC_TO_HBM2(elf.tohost_paddr);
            uint64_t tohost_val  = 0;

            while (tohost_val == 0 && !g_interrupted) {
                uart_poll(xdma);

                ssize_t n = dma_read(xdma, tohost_hbm2, &tohost_val, sizeof(tohost_val));
                if (n != sizeof(tohost_val)) {
                    fprintf(stderr, "[pegasus] Warning: DMA read tohost returned %zd\n", n);
                    tohost_val = 0;
                }

                // tohost format:
                //   bit0=1  → exit event: exit_code = tohost >> 1
                //   bit0=0  → syscall request (e.g. printf → write syscall)
                //             respond by writing 1 to fromhost
                if (tohost_val != 0 && (tohost_val & 1) == 0) {
                    // Syscall: acknowledge by writing 1 to fromhost
                    if (elf.fromhost_paddr != 0) {
                        uint64_t resp = 1;
                        uint64_t fromhost_hbm2 = SOC_TO_HBM2(elf.fromhost_paddr);
                        dma_write(xdma, fromhost_hbm2, &resp, sizeof(resp));
                    }
                    tohost_val = 0;  // Reset and keep polling
                }

                if (tohost_val == 0) {
                    usleep(1000);  // 1ms poll interval
                }
            }

            if (g_interrupted) {
                fprintf(stderr, "\n[pegasus] Interrupted by user.\n");
                scu_halt(xdma);
                xdma_close(xdma);
                return 130;
            }

            // Decode exit code
            int exit_code = 0;
            if ((tohost_val & 1) == 1) {
                exit_code = static_cast<int>((tohost_val >> 1) & 0x7FFFFFFF);
            }

            if (exit_code == 0) {
                fprintf(stderr, "[pegasus] Program exited successfully (exit code 0)\n");
            } else {
                fprintf(stderr, "[pegasus] Program exited with code %d\n", exit_code);
            }

            scu_halt(xdma);
            uint64_t cycles = scu_read_cycles(xdma);
            fprintf(stderr, "[pegasus] Total DUT cycles: %lu\n", cycles);
            xdma_close(xdma);
            return exit_code;
        }
    }

    scu_halt(xdma);
    xdma_close(xdma);
    return 0;
}
