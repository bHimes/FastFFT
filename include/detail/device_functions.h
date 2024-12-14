#ifndef __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__
#define __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__

#include "../cufftdx/include/cufftdx/include/cufftdx.hpp"
#include "../cufftdx/include/cufftdx/include/operators/direction.hpp"

namespace FastFFT {

template <class FFT>
constexpr float exp_i2pi_P( ) {
    // This is the twiddle factor for the FULL xform, which uses float(-2.0 * pi_v<double> / double(transform_size.N));
    // We only know P = N/Q at compile time, but it still makes sense to calculate this.
    if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::forward ) {
        return float(-2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value));
    }
    else {
        return float(2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value));
    }
}

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {

    ComplexType c;
    c.x = s * (a.x * b.x + a.y * b.y);
    c.y = s * (a.y * b.x - a.x * b.y);
    return c;
}

#define USEFASTSINCOS
// The __sincosf doesn't appear to be the problem with accuracy, likely just the extra additions, but it probably also is less flexible with other types. I don't see a half precision equivalent.
#ifdef USEFASTSINCOS
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    __sincosf(arg, s, c);
}
#else
__device__ __forceinline__ void SINCOS(float arg, float* s, float* c) {
    sincos(arg, s, c);
}
#endif

} // namespace FastFFT

#endif // __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__