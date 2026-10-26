#pragma once
// The norm_window RVV kernel image, embedded from this directory's build
// output.
#include <cstdint>

alignas(4) constexpr uint8_t norm_window_image[] = {
#embed "norm_window.image"
};
// Header words, see stack/kernels/rvv/kernel.ld: 1 = code bytes, 2 = entry.
constexpr uint32_t norm_window_entry =
    norm_window_image[8] | norm_window_image[9] << 8 |
    norm_window_image[10] << 16 | uint32_t(norm_window_image[11]) << 24;
constexpr uint32_t norm_window_text =
    norm_window_image[4] | norm_window_image[5] << 8 |
    norm_window_image[6] << 16 | uint32_t(norm_window_image[7]) << 24;
