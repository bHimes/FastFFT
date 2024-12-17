#include "tests.h"
#include <cufft.h>
#include <cufftXt.h>

template <int Rank>
void compare_libraries(std::vector<int> size, FastFFT::SizeChangeType::Enum size_change_type, bool do_rectangle) {

    using SCT = FastFFT::SizeChangeType::Enum;

    // bool set_padding_callback = false; // the padding callback is slower than pasting in b/c the read size of the pointers is larger than the actual data. do not use.
    bool is_size_change_decrease = false;

    if ( size_change_type == SCT::decrease ) {
        is_size_change_decrease = true;
    }

    // For an increase or decrease in size, we have to shrink the loop by one,
    // for a no_change, we don't because every size is compared to itself.
    int loop_limit = 1;
    if ( size_change_type == SCT::no_change )
        loop_limit = 0;

    if ( Rank == 3 && do_rectangle ) {
        std::cout << "ERROR: cannot do 3d and rectangle at the same time" << std::endl;
        return;
    }

    short4 input_size;
    short4 output_size;
    for ( int iSize = 0; iSize < size.size( ) - loop_limit; iSize++ ) {
        int oSize;
        int loop_size;
        // TODO: the logic here is confusing, clean it up
        if ( size_change_type != SCT::no_change ) {
            oSize     = iSize + 1;
            loop_size = size.size( );
        }
        else {
            oSize     = iSize;
            loop_size = oSize + 1;
        }

        while ( oSize < loop_size ) {

            if ( is_size_change_decrease ) {
                output_size = make_short4(size[iSize], size[iSize], 1, 0);
                input_size  = make_short4(size[oSize], size[oSize], 1, 0);
                if ( Rank == 3 ) {
                    output_size.z = size[iSize];
                    input_size.z  = size[oSize];
                }
            }
            else {
                input_size  = make_short4(size[iSize], size[iSize], 1, 0);
                output_size = make_short4(size[oSize], size[oSize], 1, 0);
                if ( Rank == 3 ) {
                    input_size.z  = size[iSize];
                    output_size.z = size[oSize];
                }
            }

            // bool test_passed = true;

            Image<float, float2> FT_input(input_size);
            Image<float, float2> FT_output(output_size);

            // We just make one instance of the FourierTransformer class, with calc type float.
            // For the time being input and output are also float. TODO caFlc optionally either fp16 or nv_bloat16, TODO inputs at lower precision for bandwidth improvement.
            FastFFT::FourierTransformer<float, float, float2, Rank> FT;
            // Create an instance to copy memory also for the cufft tests.
            FastFFT::FourierTransformer<float, float, float2, Rank> targetFT;

            float* FT_buffer;
            float* targetFT_buffer;

            if ( is_size_change_decrease ) {
                FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                FT.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);

                targetFT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, input_size.x, input_size.y, input_size.z);
                targetFT.SetInverseFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
            }
            else {
                FT.SetForwardFFTPlan(input_size.x, input_size.y, input_size.z, output_size.x, output_size.y, output_size.z);
                FT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);

                targetFT.SetForwardFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
                targetFT.SetInverseFFTPlan(output_size.x, output_size.y, output_size.z, output_size.x, output_size.y, output_size.z);
            }

            short4 fwd_dims_in  = FT.ReturnFwdInputDimensions( );
            short4 fwd_dims_out = FT.ReturnFwdOutputDimensions( );
            short4 inv_dims_in  = FT.ReturnInvInputDimensions( );
            short4 inv_dims_out = FT.ReturnInvOutputDimensions( );

            FT_input.real_memory_allocated  = FT.ReturnInputMemorySize( );
            FT_output.real_memory_allocated = FT.ReturnInvOutputMemorySize( );

            size_t device_memory = std::max(FT_input.n_bytes_allocated, FT_output.n_bytes_allocated);
            std::cerr << "device_memory: " << device_memory << std::endl;
            cudaErr(cudaMallocAsync((void**)&FT_buffer, device_memory, cudaStreamPerThread));
            cudaErr(cudaMallocAsync((void**)&targetFT_buffer, device_memory, cudaStreamPerThread));
            // Set to zero
            cudaErr(cudaMemsetAsync(FT_buffer, 0, device_memory, cudaStreamPerThread));
            cudaErr(cudaMemsetAsync(targetFT_buffer, 0, device_memory, cudaStreamPerThread));

            bool set_fftw_plan = false;
            FT_input.Allocate(set_fftw_plan);
            FT_output.Allocate(set_fftw_plan);

            // Set a unit impulse at the center of the input array.
            // For now just considering the real space image to have been implicitly quadrant swapped so the center is at the origin.
            FT.SetToConstant(FT_input.real_values, FT_input.real_memory_allocated, 0.0f);
            FT.SetToConstant(FT_output.real_values, FT_output.real_memory_allocated, 0.0f);

            // Place these values at the origin of the image and after convolution, should be at 0,0,0.
            float divide_by = FFT_DEBUG_STAGE == 0 ? 1.f : FFT_DEBUG_STAGE < 4 ? sqrtf(16.f)
                                                   : FFT_DEBUG_STAGE == 4      ? sqrtf(16.f * 16.f)
                                                   : FFT_DEBUG_STAGE < 7       ? sqrtf(16.f * 16.f * 16.f)
                                                                               : 256.;

            float counter       = 0.f;
            int   pixel_counter = 0;
            for ( int i = 0; i < 16; i++ ) {
                for ( int j = 0; j < 16; j++ ) {
                    FT_input.real_values[pixel_counter] = counter / divide_by;
                    counter += 1.f;
                    pixel_counter++;
                }
                pixel_counter += FT_input.padding_jump_value;
            }

            // Transform the target on the host prior to transfer.

            cudaErr(cudaMemcpyAsync(FT_buffer, FT_input.real_values, FT_input.n_bytes_allocated, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

            FT.FwdFFT(FT_buffer);
            // FIXME: the internal buffering scheme doesn't seem to be working as I thought when DEBUGSTAGE = 1
            FT.CopyDeviceToDeviceAndSynchronize(FT_buffer);
            FT.InvFFT(FT_buffer);

            FT.SetToConstant(FT_output.real_values, FT_output.real_memory_allocated, 0.0f);
            FT.SetToConstant(FT_input.real_values, FT_input.real_memory_allocated, 0.0f);
            bool continue_debugging;
            if ( is_size_change_decrease ) {
                // Because the output is smaller than the input, we just copy to FT input.
                // TODO: In reality, we didn't need to allocate FT_output at all in this case
                FT.CopyDeviceToHostAndSynchronize(FT_input.real_values);
                continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(FT_input, fwd_dims_in, fwd_dims_out, inv_dims_in, inv_dims_out, __LINE__);
            }
            else {
                // the output is equal or > the input, so we can always copy there.
                FT.CopyDeviceToHostAndSynchronize(FT_output.real_values);
                continue_debugging = debug_partial_fft<FFT_DEBUG_STAGE, Rank>(FT_output, fwd_dims_in, fwd_dims_out, inv_dims_in, inv_dims_out, __LINE__);
            }

            for ( int i = 0; i < 10; i++ ) {
                std::cout << FT_input.real_values[i] << " ";
            }
            MyTestPrintAndExit(continue_debugging, "Partial FFT debug stage " + std::to_string(FFT_DEBUG_STAGE));

            oSize++;
            // We don't want to loop if the size is not actually changing.
            cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
            cudaErr(cudaFree(FT_buffer));
            cudaErr(cudaFree(targetFT_buffer));
        } // while loop over pad to size

    } // for loop over pad from size
}

int main(int argc, char** argv) {

    using SCT = FastFFT::SizeChangeType::Enum;

    std::string test_name;
    // Default to running all tests
    bool run_2d_performance_tests = false;
    bool run_3d_performance_tests = false;

    const std::string_view          text_line = "simple convolution";
    std::array<std::vector<int>, 4> test_size = FastFFT::CheckInputArgs(argc, argv, text_line, run_2d_performance_tests, run_3d_performance_tests);

    // TODO: size decrease
    if ( run_2d_performance_tests ) {
#ifdef HEAVYERRORCHECKING_FFT
        std::cout << "Running performance tests with heavy error checking.\n";
        std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
// exit(1);
#endif
        std::vector<int> size = {16, 16};

        SCT size_change_type;
        // Set the SCT to no_change, increase, or decrease
        // size_change_type = SCT::no_change;
        // compare_libraries<2>(size, size_change_type, false);

        size_change_type = SCT::increase;
        compare_libraries<2>(size, size_change_type, false);

        // size_change_type = SCT::decrease;
        // compare_libraries<2>(size, size_change_type, false);
    }

#ifdef FastFFT_3d_instantiation
//     if ( run_3d_performance_tests ) {
// #ifdef HEAVYERRORCHECKING_FFT
//         std::cout << "Running performance tests with heavy error checking.\n";
//         std::cout << "This doesn't make sense as the synchronizations are invalidating.\n";
// #endif

//         SCT size_change_type;

//         size_change_type = SCT::no_change;
//         compare_libraries<3>(FastFFT::test_size_3d, size_change_type, false);

//         // TODO: These are not yet completed.
//         // size_change_type = SCT::increase;
//         // compare_libraries<3>(FastFFT::test_size, do_3d, size_change_type, false);

//         // size_change_type = SCT::decrease;
//         // compare_libraries(FastFFT::test_size, do_3d, size_change_type, false);
//     }
#endif

    return 0;
};