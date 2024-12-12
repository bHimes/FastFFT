#ifndef __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__
#define __INCLUDE_DETAIL_DEVICE_FUNCTIONS_H__

#include "../cufftdx/include/cufftdx/include/cufftdx.hpp"
#include "../cufftdx/include/cufftdx/include/operators/direction.hpp"

namespace FastFFT {

// Provides precalculated Twiddles for _exp_i2piS_div_N<FFT, Precision, Singal N, Stride S>()
// #include "compile_time_twiddles.h"

// template <typename T>
// constexpr T my_constexpr_pow2(T x, int n) {
//     return n == 0 ? 1 : n % 2 == 0 ? my_test(x * x, n / 2)
//                                    : my_test(x * x, (n - 1) / 2) * x;
// }

template <class FFT>
constexpr float _i2pi_P( ) {
    // This is the twiddle factor for the FULL xform, which uses float(-2.0 * pi_v<double> / double(transform_size.N));
    // We only know P = N/Q at compile time, but it still makes sense to calculate this.
    if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::forward ) {
        return float(-2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value));
    }
    else {
        return float(2.0 * pi_v<double> / double(cufftdx::size_of<FFT>::value));
    }
}

template <class FFT, int Q>
constexpr float _i2pi_N( ) {
    // This is the twiddle factor for the FULL xform, which uses float(-2.0 * pi_v<double> / double(transform_size.N));
    // We only know P = N/Q at compile time, but it still makes sense to calculate this.
    if constexpr ( cufftdx::direction_of<FFT>::value == cufftdx::fft_direction::forward ) {
        return float(-2.0 * pi_v<double> / (double(cufftdx::size_of<FFT>::value) * double(Q)));
    }
    else {
        return float(2.0 * pi_v<double> / (double(cufftdx::size_of<FFT>::value) * double(Q)));
    }
}

template <bool flag = false>
inline void static_non_complexmul_type( ) { static_assert(flag, "static_non_complexmul_type"); }

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {
    // Not sure if this is the best way, but it at least ensures the size is right.
    static_assert(sizeof(ComplexType) == 2 * sizeof(ScalarType), "ComplexType must be twice the size of ScalarType");
    static_assert(std::is_arithmetic_v<ScalarType>, "ScalarType must be an arithmetic type");
    ComplexType c;
    if constexpr ( std::is_same_v<ScalarType, float> ) {
        c.x = s * __fmaf_ieee_rn(a.x * b.x, -a.y * b.y);
        c.y = s * __fmaf_ieee_rn(a.y * b.x, +a.x * b.y);
    }
    else if constexpr ( std::is_same_v<ScalarType, __nv_bfloat16> || std::is_same_v<ScalarType, __half> ) {
        c.x = s * __hfma(a.x * b.x, -a.y * b.y);
        c.y = s * __hfma(a.y * b.x, +a.x * b.y);
    }
    else {
        static_non_complexmul_type( );
    }

    return c;
}

// Complex a * conj b multiplication
template <typename ComplexType, typename ScalarType>
static __device__ __host__ inline auto ComplexConjMulAndScale(const ComplexType a, const ComplexType b, ScalarType s) -> decltype(b) {

    // Not sure if this is the best way, but it at least ensures the size is right.
    static_assert(sizeof(ComplexType) == 2 * sizeof(ScalarType), "ComplexType must be twice the size of ScalarType");
    static_assert(std::is_arithmetic_v<ScalarType>, "ScalarType must be an arithmetic type");
    ComplexType c;
    if constexpr ( std::is_same_v<ScalarType, float> ) {
        c.x = s * __fmaf_ieee_rn(a.x * b.x, +a.y * b.y);
        c.y = s * __fmaf_ieee_rn(a.y * b.x, -a.x * b.y);
    }
    else if constexpr ( std::is_same_v<ScalarType, __nv_bfloat16> || std::is_same_v<ScalarType, __half> ) {
        c.x = s * __hfma(a.x * b.x, +a.y * b.y);
        c.y = s * __hfma(a.y * b.x, -a.x * b.y);
    }
    else {
        static_non_complexmul_type( );
    }

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