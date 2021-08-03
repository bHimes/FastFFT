// Insert some license stuff here

#ifndef fast_FFT_H_
#define fast_FFT_H_



// #include "/groups/himesb/git/cufftdx/example/block_io.hpp"
// #include "/groups/himesb/git/cufftdx/example/common.hpp"
// #include <iostream>

namespace FastFFT {

  constexpr const float PIf = 3.14159275358979323846f;

  typedef
	struct __align__(8) _Offsets{
    short shared_input;
    short shared_output;
    int pixel_pitch;
  } Offsets;

  Offsets mem_offsets;

class FourierTransformer 
  
{

public:

  // Used to specify input/calc/output data types
  enum DataType { int4_2, uint8, int8, uint16, int16, fp16, bf16, tf32, uint32, int32, fp32};
  enum OriginType { natural, centered, quadrant_swapped}; // Used to specify the origin of the data
  short  padding_jump_val;
  int input_memory_allocated;
  int output_memory_allocated;
  int input_number_of_real_values;
  int output_number_of_real_values;

  FourierTransformer(DataType wanted_calc_data_type);
  // FourierTransformer(const FourierTransformer &); // Copy constructor
  virtual ~FourierTransformer();

  // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
  void SetInputDimensionsAndType(size_t input_logical_x_dimension, 
                                size_t input_logical_y_dimension, 
                                size_t input_logical_z_dimension, 
                                bool is_padded_input, 
                                bool is_host_memory_pinned, 
                                DataType input_data_type,
                                OriginType input_origin_type);
  
  void SetOutputDimensionsAndType(size_t output_logical_x_dimension, 
                                  size_t output_logical_y_dimension, 
                                  size_t output_logical_z_dimension, 
                                  bool is_padded_output, 
                                  DataType output_data_type,
                                  OriginType output_origin_type);

  // void SetInputPointer(int16* input_pointer, bool is_input_on_device);
  void SetInputPointer(float* input_pointer, bool is_input_on_device);
  void CopyHostToDevice();
  void CopyDeviceToHost(bool is_in_buffer, bool free_gpu_memory, bool unpin_host_memory);
  void Deallocate();
  void UnPinHostMemory();


  // FFT calls

  // 1:1 no resizing or anything fancy.
  void SimpleFFT_NoPadding();
  void FFT_R2C_Transposed();
  void FFT_C2C_WithPadding(bool forward_transform);
  void FFT_C2C(bool forward_transform);
  void FFT_C2R_Transposed();






  inline int ReturnPaddedMemorySize(short4 & wanted_dims) 
  {
    int wanted_memory = 0;
    if (wanted_dims.x % 2 == 0) { padding_jump_val = 2; wanted_memory = wanted_dims.x / 2 + 1;}
    else { padding_jump_val = 1 ; wanted_memory = (wanted_dims.x - 1) / 2 + 1;}

    wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
    wanted_memory *= 2; // room for complex
    wanted_dims.w = (wanted_dims.x + padding_jump_val) / 2; // number of complex elements in the X dimesnions after FFT.
    return wanted_memory;
  };

  template<typename T, bool is_on_host = true>
  void SetToConstant(T* input_pointer, int N_values, const T wanted_value)
  {
    if (is_on_host) 
    {
      for (int i = 0; i < N_values; i++)
      {
        input_pointer[i] = (T)wanted_value;
      }
    }
    else
    {
      exit(-1);
    }
  }
private:


  DataType input_date_type;
  DataType calc_data_type;
  DataType output_data_type;

  OriginType input_origin_type;
  OriginType output_origin_type;

  // booleans to track state, could be bit fields but that seem opaque to me.
  bool is_in_memory_host_pointer;
  bool is_in_memory_device_pointer;
  bool is_in_buffer_memory;

  bool is_host_memory_pinned;

  bool is_fftw_padded_input;
  bool is_fftw_padded_output;
  bool is_fftw_padded_buffer;

  bool is_set_input_params;
  bool is_set_output_params;

  short4 dims_in;
  short4 dims_out;
  short  fft_status; // 


  dim3 gridDims;
  dim3 threadsPerBlock;

  float* host_pointer;
  float* pinnedPtr;
  float* device_pointer_fp32; float2* device_pointer_fp32_complex;
  float* buffer_fp32; float2* buffer_fp32_complex;
  __half* device_pointer_fp16; __half2* device_pointer_fp16_complex;

  float twiddle_in; // (twiddle factor for input)
  int   Q; // N/L (FULL SIZE/ NON-ZERO DIMENSIONS)


  void SetDefaults();
  inline void SetLaunchParameters(const int& ept)
  {
    
    switch (fft_status)
    {
      case 0:
        threadsPerBlock = dim3(dims_in.x/ept, 1, 1);
        gridDims = dim3(1, dims_in.y, 1); 
        mem_offsets.shared_input = dims_in.x;
        mem_offsets.shared_output = dims_in.w*2;
        mem_offsets.pixel_pitch = dims_out.y;
        twiddle_in = -2*PIf/dims_out.x;
        Q = dims_out.x / dims_in.x; 
        break;
      case 1:
        threadsPerBlock = dim3(dims_in.y/ept, 1, 1); 
        gridDims = dim3(1, dims_out.w, 1);
        mem_offsets.shared_input = dims_in.y;
        mem_offsets.shared_output = dims_out.y;
        mem_offsets.pixel_pitch = dims_out.y;
        twiddle_in = -2*PIf/dims_out.y;
        Q = dims_out.y / dims_in.y; // FIXME assuming for now this is already divisible
        break;
      case 2:
        threadsPerBlock = dim3(dims_out.y/ept, 1, 1);
        gridDims = dim3(1, dims_out.w, 1);
        twiddle_in = -2*PIf/dims_out.y;
        Q = 1; // Already full size - FIXME when working out limited number of output pixels       
        mem_offsets.shared_input = 0;
        mem_offsets.shared_output = 0;
        mem_offsets.pixel_pitch = dims_out.y;
        break;
      case 3:
        threadsPerBlock = dim3(dims_out.w/ept, 1, 1); // or w*2?
        gridDims = dim3(1, dims_out.y, 1);
        twiddle_in = -2*PIf/dims_out.y;
        Q = 1; // Already full size - FIXME when working out limited number of output pixels  
        mem_offsets.shared_input = 0;
        mem_offsets.shared_output = dims_out.w*2; // It turns out that we need pitch in and pitch out, FIXME hack
        mem_offsets.pixel_pitch = dims_out.y;         
        break;
      default:
        std::cerr << "ERROR: Unrecognized fft_status" << std::endl;
        exit(-1);
        
    }

  }



};




} // namespace fast_FFT



#endif