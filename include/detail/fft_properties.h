#ifndef __INCLUDE_DETAIL_FFT_PROPERTIES_H__
#define __INCLUDE_DETAIL_FFT_PROPERTIES_H__

namespace FastFFT {

typedef struct __align__(16) _FFT_Size {
    // Following Sorensen & Burrus 1993 for clarity
    unsigned int N; // N : 1d FFT size
    unsigned int L; // L : number of non-zero output/input points
    unsigned int P; // P >= L && N % P == 0 : The size of the sub-FFT used to compute the full transform. Currently also must be a power of 2.
    unsigned int Q; // Q = N/P : The number of sub-FFTs used to compute the full transform
}

FFT_Size;

typedef struct __align__(16) _Offsets {
    unsigned int shared_input;
    unsigned int shared_output;
    unsigned int physical_x_input;
    unsigned int physical_x_output;
}

Offsets;

typedef struct __align__(64) _LaunchParams {
    FFT_Size transform_size;
    Offsets  mem_offsets;
    dim3     gridDims;
    dim3     threadsPerBlock;
}

LaunchParams;

} // namespace FastFFT

#endif