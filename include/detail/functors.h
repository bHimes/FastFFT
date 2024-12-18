#ifndef __INCLUDE_DETAILS_FUNCTORS_H__
#define __INCLUDE_DETAILS_FUNCTORS_H__

#include "concepts.h"

// TODO: doc and namespace
// FIXME: the is_final is a bit of a hack to make sure we can tell if the functor is a NOOP or not

namespace FastFFT {

namespace KernelFunction {

// Maybe a better way to check , but using keyword final to statically check for non NONE types
// TODO: is marking the operatior()() inline sufficient or does the struct need to be marked inline as well?
template <typename T, IKF_t ikf, typename = void>
struct my_functor {};

// At times, we want to specifically pass the noop functor, which will
// be ignored because it is (NOT) marked final
template <typename T>
struct my_functor<T, IKF_t::NOOP> {
    __device__ __forceinline__ void
    operator( )( ){ };
};

// So we can generically call a functor from somewhere else
// but have empty code for noop case
template <class FunctorType, typename... Args>
__device__ __host__ __forceinline__ void callFunctor(FunctorType&& f, Args&&... args) {
    if constexpr ( ! std::is_same_v<std::decay_t<FunctorType>, default_functor_noop_t> )
        f(std::forward<Args>(args)...);
}

// EnableIf<IsAllowedComplexBaseType<T>>
//struct my_functor<T, 4, IKF_t::CONJ_MUL, EnableIf<IsAllowedComplexBaseType<T>>> final {

template <bool flag = false>
inline void static_assert_CONJ_MUL_type_not_found( ) { static_assert(flag, "static_assert_CONJ_MUL_type_not_found"); }

template <typename T>
struct my_functor<T, IKF_t::CONJ_MUL, EnableIf<IsAllowedInputType<T>>> final {

    using other_t = std::conditional_t<std::is_same_v<T, float>, __half,
                                       std::conditional_t<std::is_same_v<T, float2>, __half2, void>>;
    static_assert(! std::is_same_v<other_t, void>, "static_assert_CONJ_MUL_ other type_not_found");

    using base_t = std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, float2>, float, void>;
    static_assert(! std::is_same_v<base_t, void>, "static_assert_CONJ_MUL_base type_not_found");

    base_t tmp;

    __device__ __forceinline__ void
    operator( )(T& template_fft, const other_t& target_fft) {
        if constexpr ( std::is_same_v<T, float2> ) {

            if constexpr ( std::is_same_v<other_t, __half2> ) {
                // TODO: can we just .x and .y the half2?
                tmp            = (template_fft.x * __low2float(target_fft) + template_fft.y * __high2float(target_fft));
                template_fft.y = (template_fft.x * __high2float(target_fft) - template_fft.y * __low2float(target_fft.x));
                template_fft.x = tmp;
            }
            else {
                tmp            = (template_fft.x * target_fft.x + template_fft.y * target_fft.y);
                template_fft.y = (template_fft.x * target_fft.y - template_fft.y * target_fft.x);
                template_fft.x = tmp;
                return tmp;
            }
        }
        else
            static_assert_CONJ_MUL_type_not_found( );
    }

    __device__ __forceinline__ void
    operator( )(T& template_fft_x, T& template_fft_y, const other_t& target_fft_x, const other_t& target_fft_y) {
        if constexpr ( IsAllowedRealType<T> ) {

            if constexpr ( std::is_same_v<other_t, __half> ) {
                tmp            = (template_fft_x * target_fft_x + template_fft_y * target_fft_y);
                template_fft_y = (template_fft_x * target_fft_y - template_fft_y * target_fft_x);
                template_fft_x = tmp;
            }
            else {
                tmp            = (template_fft_x * __half2float(target_fft_x) + template_fft_y * __half2float(target_fft_y));
                template_fft_y = (template_fft_x * __half2float(target_fft_y) - template_fft_y * __half2float(target_fft_x));
                template_fft_x = tmp;
            }
        }
        else
            static_assert_CONJ_MUL_type_not_found( );
    }
};

template <typename T>
struct my_functor<T, IKF_t::CONJ_MUL_THEN_SCALE, EnableIf<IsAllowedRealType<T>>> final {

    // Pass in the scale factor on construction
    my_functor(const T& scale_factor) : scale_factor(scale_factor) {}

    __device__ __forceinline__ void
    operator( )(T& template_fft_x, T& template_fft_y, const T& target_fft_x, const T& target_fft_y) {
        // Is there a better way than declaring this variable each time?
        // This is target * conj(template)
        T tmp          = (template_fft_x * target_fft_x + template_fft_y * target_fft_y) * scale_factor;
        template_fft_y = (template_fft_x * target_fft_y - template_fft_y * target_fft_x) * scale_factor;
        template_fft_x = tmp;
    }

  private:
    const T scale_factor;
};

template <bool flag = false>
inline void static_assert_invalid_IKF_t_SCALE_arg( ) { static_assert(flag, "static_assert_invalid_IKF_t_SCALE_arg"); }

template <typename T>
struct my_functor<T, IKF_t::SCALE, EnableIf<IsAllowedInputType<T>>> final {

    const T scale_factor;

    // Pass in the scale factor on construction
    __device__ __forceinline__ my_functor(const T& scale_factor) : scale_factor(scale_factor) {}

    __device__ __forceinline__ T
    operator( )(T input_value) {
        if constexpr ( std::is_same_v<T, __half> || std::is_same_v<T, float> )
            return input_value *= scale_factor;
        else if constexpr ( std::is_same_v<T, __half2> || std::is_same_v<T, float2> ) {
            input_value.x *= scale_factor;
            input_value.y *= scale_factor;
            return input_value;
        }
        else
            static_assert_invalid_IKF_t_SCALE_arg( );
    }
};

} // namespace KernelFunction
} // namespace FastFFT

#endif