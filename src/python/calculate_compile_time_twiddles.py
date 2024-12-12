#!/usr/bin/python3

import numpy as  np
# These are used for the factor exp(-i2pi / N * FFT::stride)

# We currently limit the size of FFTs to be power of 2 from 2^4 to 2^12 (16 to 4096)
# We currently limit the ept to be a power of 2 from 2^2 to 2^8 (2 to 6)

# The stride is the FFT size divided by the ept size

# In principle the Signal length doesn't need to match the FFT size, but we do minimally resitrict it to be even

# test with float
output_file = "compile_time_twiddles.h"
with open(output_file, "w") as f:
    
    print(f"#ifndef COMPILE_TIME_TWIDDLES_H", file=f)
    print(f"#define COMPILE_TIME_TWIDDLES_H\n", file=f)

    print(f"template <bool flag = false>",file=f)
    print(f"inline void static_no_matching_twiddle_specialization( ) {{ static_assert(flag, \"static_no_matching_twiddle_specialization\"); }}\n",file=f)
    print(f"template<class FFT, typename Precision, unsigned int Size /* FFT size */, unsigned int Stride /* FFT stride */>()", file=f)
    print(f"constexpr Precision _exp_i2piS_div_N(){{\n    static_no_matching_twiddle_specialization();\n}};\n\n", file=f)
 
# loop over possible FFT sizes
f = open(output_file, "a")

n=0
for signal_length in range(2**4, 2**12 + 2, 2):
    used_stride = []
    for fft_size in range(4, 13):
        d_fft_size = np.double(2**fft_size)
        # loop over possible ept sizes
        for ept_size in range(2, 9):
            if ept_size > fft_size:
                continue

            # calculate the stride
            stride = 2**fft_size // 2**ept_size

            if stride in used_stride:
                continue
            used_stride.append(stride)
            d_stride = np.double(stride)
            
            # calculate the twiddle factor in double precision
            twiddle_c = np.exp(-1j * 2.0 * np.pi / d_fft_size * d_stride).astype(np.complex64)
            n+=1
            print(f"n: {n} signal_length: {signal_length}, fft_size: {2**fft_size}, ept_size: {2**ept_size}, stride: {stride}, twiddle_c: {twiddle_c.real:4.26f} + {twiddle_c.imag:4.26f}")

            print(f"template<>\nconstexpr Precision _exp_i2piS_div_N<class FFT, typename Precision, {signal_length}, {stride}>(){{\n    if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::forward )\n        return Precision{{{twiddle_c.real:4.26f},{twiddle_c.imag:4.26f}}}\n    else\n        return Precision{{{twiddle_c.real:4.26f},{-twiddle_c.imag:4.26f}}};\n}};\n", file=f)

print(f"#endif",file=f)
f.close()
