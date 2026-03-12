#include "pegasus.h"

#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

// libelf headers
#include <gelf.h>
#include <libelf.h>

// Parse an ELF file and return load segment info + tohost/fromhost symbol addresses
ELFInfo elf_parse(const char *elf_path) {
    ELFInfo info;
    info.path         = elf_path;
    info.entry        = 0;
    info.tohost_paddr = 0;
    info.fromhost_paddr = 0;

    if (elf_version(EV_CURRENT) == EV_NONE) {
        fprintf(stderr, "[pegasus] libelf init failed: %s\n", elf_errmsg(-1));
        exit(1);
    }

    int fd = open(elf_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[pegasus] Cannot open ELF: %s: %s\n",
                elf_path, strerror(errno));
        exit(1);
    }

    Elf *elf = elf_begin(fd, ELF_C_READ, nullptr);
    if (!elf) {
        fprintf(stderr, "[pegasus] elf_begin failed: %s\n", elf_errmsg(-1));
        close(fd);
        exit(1);
    }

    if (elf_kind(elf) != ELF_K_ELF) {
        fprintf(stderr, "[pegasus] %s is not an ELF file\n", elf_path);
        elf_end(elf);
        close(fd);
        exit(1);
    }

    GElf_Ehdr ehdr;
    if (!gelf_getehdr(elf, &ehdr)) {
        fprintf(stderr, "[pegasus] gelf_getehdr failed: %s\n", elf_errmsg(-1));
        elf_end(elf);
        close(fd);
        exit(1);
    }

    if (ehdr.e_machine != EM_RISCV) {
        fprintf(stderr, "[pegasus] Warning: ELF machine type is not RISC-V (got %u)\n",
                ehdr.e_machine);
    }

    info.entry = ehdr.e_entry;

    // --- Walk program headers to find PT_LOAD segments ---
    size_t phnum = 0;
    elf_getphdrnum(elf, &phnum);

    for (size_t i = 0; i < phnum; i++) {
        GElf_Phdr phdr;
        if (!gelf_getphdr(elf, static_cast<int>(i), &phdr)) continue;
        if (phdr.p_type != PT_LOAD)                          continue;
        if (phdr.p_filesz == 0)                              continue;

        ELFSegment seg;
        seg.paddr  = phdr.p_paddr;
        seg.vaddr  = phdr.p_vaddr;
        seg.filesz = phdr.p_filesz;
        seg.memsz  = phdr.p_memsz;
        seg.offset = phdr.p_offset;
        info.segments.push_back(seg);
    }

    // --- Walk symbol table for tohost / fromhost ---
    Elf_Scn *scn = nullptr;
    while ((scn = elf_nextscn(elf, scn)) != nullptr) {
        GElf_Shdr shdr;
        gelf_getshdr(scn, &shdr);
        if (shdr.sh_type != SHT_SYMTAB && shdr.sh_type != SHT_DYNSYM) continue;

        Elf_Data *data = elf_getdata(scn, nullptr);
        size_t sym_count = (data && shdr.sh_entsize) ?
                           data->d_size / shdr.sh_entsize : 0;

        for (size_t j = 0; j < sym_count; j++) {
            GElf_Sym sym;
            gelf_getsym(data, static_cast<int>(j), &sym);
            const char *name = elf_strptr(elf, shdr.sh_link, sym.st_name);
            if (!name) continue;
            if (strcmp(name, "tohost") == 0) {
                info.tohost_paddr = sym.st_value;
            } else if (strcmp(name, "fromhost") == 0) {
                info.fromhost_paddr = sym.st_value;
            }
        }
    }

    elf_end(elf);
    close(fd);

    if (info.segments.empty()) {
        fprintf(stderr, "[pegasus] Warning: No PT_LOAD segments found in %s\n", elf_path);
    }
    if (info.tohost_paddr == 0) {
        fprintf(stderr, "[pegasus] Warning: 'tohost' symbol not found in %s\n", elf_path);
    }

    return info;
}

// Load all PT_LOAD segments into FPGA HBM2 via DMA
// Address translation: SoC physical → HBM2 DMA address (subtract SOC_DRAM_BASE)
void elf_load(PegasusXDMA *xdma, const ELFInfo *info) {
    // Open ELF file for raw reads
    int fd = open(info->path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[pegasus] Cannot open ELF for loading: %s: %s\n",
                info->path.c_str(), strerror(errno));
        exit(1);
    }

    for (const auto &seg : info->segments) {
        // Translate SoC physical address to HBM2 DMA address
        if (seg.paddr < SOC_DRAM_BASE) {
            fprintf(stderr, "[pegasus] Segment at 0x%" PRIx64 " is below SOC_DRAM_BASE "
                            "(0x%" PRIx64 ") — skipping\n",
                    seg.paddr, (uint64_t)SOC_DRAM_BASE);
            continue;
        }
        uint64_t hbm2_addr = SOC_TO_HBM2(seg.paddr);

        fprintf(stderr, "[pegasus] Loading segment: SoC 0x%08" PRIx64
                        " -> HBM2 0x%09" PRIx64 " (%" PRIu64 " bytes)\n",
                seg.paddr, hbm2_addr, seg.filesz);

        // Read segment from ELF file
        std::vector<uint8_t> buf(seg.filesz);
        ssize_t n = pread(fd, buf.data(), seg.filesz,
                          static_cast<off_t>(seg.offset));
        if (n != static_cast<ssize_t>(seg.filesz)) {
            fprintf(stderr, "[pegasus] Short read from ELF (got %zd, expected %" PRIu64 ")\n",
                    n, seg.filesz);
            close(fd);
            exit(1);
        }

        // DMA write to HBM2
        ssize_t written = dma_write(xdma, hbm2_addr, buf.data(), seg.filesz);
        if (written != static_cast<ssize_t>(seg.filesz)) {
            fprintf(stderr, "[pegasus] DMA write failed at HBM2 0x%09" PRIx64 ": "
                            "wrote %zd of %" PRIu64 " bytes\n",
                    hbm2_addr, written, seg.filesz);
            close(fd);
            exit(1);
        }

        // Zero-fill remaining memsz (BSS region)
        if (seg.memsz > seg.filesz) {
            size_t bss_size = seg.memsz - seg.filesz;
            std::vector<uint8_t> zeros(bss_size, 0);
            uint64_t bss_addr = hbm2_addr + seg.filesz;
            ssize_t w = dma_write(xdma, bss_addr, zeros.data(), bss_size);
            if (w != static_cast<ssize_t>(bss_size)) {
                fprintf(stderr, "[pegasus] BSS zero-fill failed at HBM2 0x%09" PRIx64 "\n",
                        bss_addr);
            }
        }
    }

    close(fd);
    fprintf(stderr, "[pegasus] ELF loaded: entry=0x%" PRIx64 ", tohost=0x%" PRIx64 "\n",
            info->entry, info->tohost_paddr);
}
