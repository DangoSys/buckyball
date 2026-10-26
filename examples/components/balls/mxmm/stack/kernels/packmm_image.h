#pragma once
// The packmm RVV kernel image, embedded from this directory's build output.
#include <cstdint>

alignas(4) constexpr uint8_t packmm_image[] = {
#embed "packmm.image"
};
// Header words, see stack/kernels/rvv/kernel.ld: 1 = code bytes, 2 = entry.
constexpr uint32_t packmm_entry = packmm_image[8] | packmm_image[9] << 8 |
                                  packmm_image[10] << 16 |
                                  uint32_t(packmm_image[11]) << 24;
constexpr uint32_t packmm_text = packmm_image[4] | packmm_image[5] << 8 |
                                 packmm_image[6] << 16 |
                                 uint32_t(packmm_image[7]) << 24;
