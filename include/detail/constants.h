#ifndef __INCLUDE_DETAIL_CONSTANTS_H__
#define __INCLUDE_DETAIL_CONSTANTS_H__

namespace FastFFT {

template <unsigned int V>
struct elements_per_thread_default {
    static constexpr unsigned int value =
            V == 16 ? 4 : V == 32 ? 8
                  : V == 64       ? 8
                  : V == 128      ? 8
                  : V == 256      ? 8
                  : V == 512      ? 8
                  : V == 1024     ? 8
                  : V == 2048     ? 8
                  : V == 4096     ? 16
                  : V == 8192     ? 16
                                  : 0;
};

template <unsigned int V>
static constexpr unsigned int elements_per_thread_default_v = elements_per_thread_default<V>::value;

} // namespace FastFFT

#endif