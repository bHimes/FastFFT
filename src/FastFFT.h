// Insert some license stuff here

#ifndef fast_FFT_H_
#define fast_FFT_H_



// #include "/groups/himesb/git/cufftdx/example/block_io.hpp"
// #include "/groups/himesb/git/cufftdx/example/common.hpp"
// #include <iostream>

namespace FastFFT {

    // For debugging

  inline void PrintVectorType(int3 input) 
  {
    std::cout << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
  }
  inline void PrintVectorType(int4 input) 
  {
    std::cout << "(x,y,z,w) " << input.x << " " << input.y << " " << input.z << " " << input.w << std::endl;
  }
    inline void PrintVectorType(dim3 input) 
  {
    std::cout << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
  }
  inline void PrintVectorType(short3 input) 
  {
    std::cout << "(x,y,z) " << input.x << " " << input.y << " " << input.z << std::endl;
  }
  inline void PrintVectorType(short4 input) 
  {
    std::cout << "(x,y,z,w) " << input.x << " " << input.y << " " << input.z << " " << input.w << std::endl;
  }

  constexpr const float PIf = 3.14159275358979323846f;

  typedef
  struct __align__(32) _DeviceProps {
    int device_id;
    int device_arch;
    int max_shared_memory_per_block;
    int max_shared_memory_per_SM;
    int max_registers_per_block;
    int max_persisting_L2_cache_size;
  } DeviceProps;

  typedef 
  struct __align__(8) _FFT_Size {
    // Following Sorensen & Burrus 1993 for clarity
    short N; // N : 1d FFT size
    short L; // L : number of non-zero output/input points 
    short P; // P >= L && N % P == 0 : The size of the sub-FFT used to compute the full transform. Currently also must be a power of 2.
    short Q; // Q = N/P : The number of sub-FFTs used to compute the full transform
  } FFT_Size;

  typedef
	struct __align__(8) _Offsets{
    short shared_input;
    short shared_output;
    short pixel_pitch_input;
    short pixel_pitch_output;
  } Offsets;

  typedef 
  struct __align__(64) _LaunchParams{
    int Q;
    float twiddle_in;
    dim3 gridDims;
    dim3 threadsPerBlock;
    Offsets mem_offsets;
  } LaunchParams;

  

  template<typename I, typename C>
  struct DevicePointers {
    // Use this to catch unsupported input/ compute types and throw exception.
    int* position_space = nullptr;
    int* position_space_buffer = nullptr;
    int* momentum_space = nullptr;
    int* momentum_space_buffer = nullptr;
    int* image_to_search = nullptr;
  };

  // Input real, compute single-precision
  template<>
  struct DevicePointers<float*, float*> {
    float*  position_space;
    float*  position_space_buffer;
    float2* momentum_space;
    float2* momentum_space_buffer;
    float2* image_to_search;

  };

  // Input real, compute half-precision FP16
  template<>
  struct DevicePointers<__half*, __half*> {
    __half*  position_space;
    __half*  position_space_buffer;
    __half2* momentum_space;
    __half2* momentum_space_buffer;
    __half2* image_to_search;
  };

  // Input complex, compute single-precision
  template<>
  struct DevicePointers<float2*, float*> {
    float2*  position_space;
    float2*  position_space_buffer;
    float2*  momentum_space;
    float2*  momentum_space_buffer;
    float2*  image_to_search;
  };

  // Input complex, compute half-precision FP16
  template<>
  struct DevicePointers<__half2*, __half*> {
    __half2*  position_space;
    __half2*  position_space_buffer;
    __half2*  momentum_space;
    __half2*  momentum_space_buffer;
    __half2*  image_to_search;

  };


  

template <class ComputeType = float, class InputType = float, class OutputType = float>
class FourierTransformer {

public:

  // Used to specify input/calc/output data types
  enum DataType { int4_2, uint8, int8, uint16, int16, fp16, bf16, tf32, uint32, int32, fp32};
  std::vector<std::string> DataTypeName { "int4_2", "uint8", "int8", "uint16", "int16", "fp16", "bf16", "tf32", "uint32", "int32", "fp32" };
  enum OriginType { natural, centered, quadrant_swapped}; // Used to specify the origin of the data
  std::vector<std::string> OriginTypeName { "natural", "centered", "quadrant_swapped" };

  short padding_jump_val;
  int   input_memory_allocated;
  int   fwd_output_memory_allocated;
  int   inv_output_memory_allocated;
  int   compute_memory_allocated;
  int   memory_size_to_copy;



  ///////////////////////////////////////////////
  // Initialization functions
  ///////////////////////////////////////////////

  FourierTransformer();
  // FourierTransformer(const FourierTransformer &); // Copy constructor
  virtual ~FourierTransformer();

  // This is pretty similar to an FFT plan, I should probably make it align with CufftPlan
  void SetForwardFFTPlan(size_t input_logical_x_dimension,  size_t input_logical_y_dimension,  size_t input_logical_z_dimension, 
                         size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                         bool is_padded_output, 
                         bool is_host_memory_pinned,
                         OriginType output_origin_type);

  void SetInverseFFTPlan(size_t input_logical_x_dimension,  size_t input_logical_y_dimension,  size_t input_logical_z_dimension, 
                         size_t output_logical_x_dimension, size_t output_logical_y_dimension, size_t output_logical_z_dimension,
                         bool is_padded_output, 
                         OriginType output_origin_type);


  // For the time being, the caller is responsible for having the memory allocated for any of these input/output pointers.
  void SetInputPointer(InputType* input_pointer, bool is_input_on_device);


  ///////////////////////////////////////////////
  // Public actions:
  // ALL public actions should call ::CheckDimensions() to ensure the meta data are properly intialized.
  // this ensures the prior three methods have been called and are valid.
  ///////////////////////////////////////////////

  void CopyHostToDevice();
  // By default we are blocking with a stream sync until complete for simplicity. This is overkill and should FIXME.
  // If int n_elements_to_copy = 0 the appropriate size will be determined by the state of the transform completed (none, fwd, inv.) 
  // For partial increase/decrease transforms, needed for testing, this will be invalid, so specify the int n_elements_to_copy.
  void CopyDeviceToHost(bool free_gpu_memory, bool unpin_host_memory, int n_elements_to_copy = 0);
  // When the size changes, we need a new host pointer
  void CopyDeviceToHost(OutputType* output_pointer, bool free_gpu_memory = true, bool unpin_host_memory = true, int n_elements_to_copy = 0);
  // FFT calls

  void FwdFFT(bool swap_real_space_quadrants = false, bool transpose_output = true);
  void InvFFT(bool transpose_output = true);
  void CrossCorrelate(float2* image_to_search, bool swap_real_space_quadrants);
  void CrossCorrelate(__half2* image_to_search, bool swap_real_space_quadrants);

  void ClipIntoTopLeft();
  void ClipIntoReal(int wanted_coordinate_of_box_center_x, int wanted_coordinate_of_box_center_y, int wanted_coordinate_of_box_center_z);

  // For all real valued inputs, assumed for any InputType that is not float2 or __half2

  int inline ReturnInputMemorySize() { return input_memory_allocated; }
  int inline ReturnFwdOutputMemorySize() { return fwd_output_memory_allocated; }
  int inline ReturnInvOutputMemorySize() { return inv_output_memory_allocated; }

  short4 inline ReturnFwdInputDimensions() { return fwd_dims_in; }
  short4 inline ReturnFwdOutputDimensions() { return fwd_dims_out; }
  short4 inline ReturnInvInputDimensions() { return fwd_dims_in; }
  short4 inline ReturnInvOutputDimensions() { return fwd_dims_out; }

  template<typename T, bool is_on_host = true>
  void SetToConstant(T* input_pointer, int N_values, const T& wanted_value)
  {
    if (is_on_host) 
    {
      for (int i = 0; i < N_values; i++)
      {
        input_pointer[i] = wanted_value;
      }
    }
    else
    {
      exit(-1);
    }
  }

  // Input is real or complex inferred from InputType
  DevicePointers<InputType*, ComputeType*> d_ptr;

   void PrintState()
  {
    std::cout << "================================================================" << std::endl; 
    std::cout << "Device Properties: " << std::endl;
    std::cout << "================================================================" << std::endl; 

    std::cout << "Device idx: " << device_properties.device_id << std::endl;
    std::cout << "max_shared_memory_per_block: " << device_properties.max_shared_memory_per_block << std::endl;
    std::cout << "max_shared_memory_per_SM: " << device_properties.max_shared_memory_per_SM << std::endl;
    std::cout << "max_registers_per_block: " << device_properties.max_registers_per_block << std::endl;
    std::cout << "max_persisting_L2_cache_size: " << device_properties.max_persisting_L2_cache_size << std::endl;
    std::cout << std::endl;

    std::cout << "State Variables:\n" << std::endl;
    std::cout << "is_in_memory_host_pointer " << is_in_memory_host_pointer << std::endl;
    std::cout << "is_in_memory_device_pointer " << is_in_memory_device_pointer << std::endl;
    std::cout << "is_in_buffer_memory " << is_in_buffer_memory << std::endl;
    std::cout << "is_host_memory_pinned " << is_host_memory_pinned << std::endl;
    std::cout << "is_fftw_padded_input " << is_fftw_padded_input << std::endl;
    std::cout << "is_fftw_padded_output " << is_fftw_padded_output << std::endl;
    std::cout << "is_real_valued_input " << is_real_valued_input << std::endl;
    std::cout << "is_set_input_params " << is_set_input_params << std::endl;
    std::cout << "is_set_output_params " << is_set_output_params << std::endl;
    std::cout << "is_size_validated " << is_size_validated << std::endl;
    std::cout << "is_set_input_pointer " << is_set_input_pointer << std::endl;
    std::cout << std::endl;

    std::cout << "Size variables:\n" << std::endl;
    std::cout << "transform_size.N " << transform_size.N << std::endl;
    std::cout << "transform_size.L " << transform_size.L << std::endl;
    std::cout << "transform_size.P " << transform_size.P << std::endl;
    std::cout << "transform_size.Q " << transform_size.Q << std::endl;
    std::cout << "fwd_dims_in.x,y,z "; PrintVectorType(fwd_dims_in); std::cout << std::endl;
    std::cout << "fwd_dims_out.x,y,z " ; PrintVectorType(fwd_dims_out); std::cout<< std::endl;
    std::cout << "inv_dims_in.x,y,z " ; PrintVectorType(inv_dims_in); std::cout<< std::endl;
    std::cout << "inv_dims_out.x,y,z " ; PrintVectorType(inv_dims_out); std::cout<< std::endl;
    std::cout << std::endl;

    std::cout << "Misc:\n" << std::endl;
    std::cout << "compute_memory_allocated " << compute_memory_allocated << std::endl;
    std::cout << "memory size to copy " << memory_size_to_copy << std::endl;
    std::cout << "fwd_size_change_type " << SizeChangeName[fwd_size_change_type] << std::endl;
    std::cout << "inv_size_change_type " << SizeChangeName[inv_size_change_type] << std::endl;
    std::cout << "transform stage complete " << TransformStageCompletedName[transform_stage_completed] << std::endl;
    std::cout << "input_origin_type " << OriginTypeName[input_origin_type] << std::endl;
    std::cout << "output_origin_type " << OriginTypeName[output_origin_type] << std::endl;
    
  }; // PrintState()

private:

  DeviceProps device_properties;
  OriginType input_origin_type;
  OriginType output_origin_type;

  // booleans to track state, could be bit fields but that seem opaque to me.
  bool is_in_memory_host_pointer; // To track allocation of host side memory
  bool is_in_memory_device_pointer; // To track allocation of device side memory.
  bool is_in_buffer_memory; // To track whether the current result is in dev_ptr.position_space or dev_ptr.position_space_buffer (momemtum space/ momentum space buffer respectively.)


  bool is_host_memory_pinned; // Specified in the constructor. Assuming host memory won't be pinned for many applications.

  bool is_fftw_padded_input; // Padding for in place r2c transforms
  bool is_fftw_padded_output; // Currently the output state will match the input state, otherwise it is an error.

  bool is_real_valued_input; // This is determined by the input type. If it is a float2 or __half2, then it is assumed to be a complex valued input function.

  bool is_set_input_params; // Yes, yes, "are" set.
  bool is_set_output_params;
  bool is_size_validated; // Defaults to false, set after both input/output dimensions are set and checked.
  bool is_set_input_pointer; // May be on the host of the device.



  int  transform_dimension; // 1,2,3d.
  FFT_Size transform_size; 
  // FIXME this seems like a bad idea. Added due to conflicing labels in switch statements, even with explicitly scope. 
  enum  SizeChangeType : uint8_t { increase , decrease , no_change  }; // Assumed to be the same for all dimesnions. This may be relaxed later.
  std::vector<std::string> SizeChangeName { "increase", "decrease", "no_change" };
  enum  TransformStageCompleted : uint8_t  { none = 10 , fwd = 11, inv = 12 };  // none must be greater than number of sizeChangeTypes, padding must match in TransformStageCompletedName vector
  std::vector<std::string> TransformStageCompletedName { "","","","","", // padding of 5
                                                         "","","","","", // padding of 5
                                                         "none", "fwd", "inv" };

  enum  DimensionCheckType : uint8_t  { CopyFromHost, CopyToHost, FwdTransform, InvTransform }; 
  std::vector<std::string> DimensionCheckName { "CopyFromHost", "CopyToHost", "FwdTransform", "InvTransform" };


  SizeChangeType fwd_size_change_type;
  SizeChangeType inv_size_change_type;

  TransformStageCompleted transform_stage_completed;

  // dims_in may change during calculation, depending on padding, but is reset after each call.
  short4 dims_in;
  short4 dims_out;
  
  short4 fwd_dims_in;
  short4 fwd_dims_out;
  short4 inv_dims_in;
  short4 inv_dims_out;

  InputType* host_pointer;
  InputType* pinnedPtr;

  void Deallocate();
  void UnPinHostMemory();

  void SetDefaults();
  void ValidateDimensions();
  void SetDimensions(DimensionCheckType check_op_type);


  enum KernelType { r2c_decomposed, // Thread based, full length.
                    r2c_decomposed_transposed, // Thread based, full length, transposed.
                    r2c_none, r2c_decrease, r2c_increase,
                    c2c_fwd_none, c2c_fwd_decrease, c2c_fwd_increase,     
                    c2c_inv_none, c2c_inv_decrease, c2c_inv_increase,                       
                    c2c_decomposed,
                    c2r_decomposed, 
                    c2r_decomposed_transposed, 
                    c2r_none, c2r_decrease, c2r_increase,
                    xcorr_increase, 
                    xcorr_decomposed }; // Used to specify the origin of the data
  // WARNING this is flimsy and prone to breaking, you must ensure the order matches the KernelType enum.
  std::vector<std::string> 
        KernelName{ "r2c_decomposed", 
                    "r2c_decomposed_transposed", 
                    "r2c_none", "r2c_decrease", "r2c_increase",
                    "c2c_fwd_none", "c2c_fwd_increase", "c2c_fwd_increase",
                    "c2c_inv_none", "c2c_inv_increase", "c2c_inv_increase",
                    "c2c_decomposed", 
                    "c2r_decomposed", 
                    "c2r_decomposed_transposed", 
                    "c2r_none", "c2r_decrease", "c2r_increase",
                    "xcorr_increase", 
                    "xcorr_decomposed" };

  inline bool IsThreadType(KernelType kernel_type)
  {
    if ( kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
         kernel_type == c2c_decomposed ||
         kernel_type == c2r_decomposed || kernel_type == c2r_decomposed_transposed || kernel_type == xcorr_decomposed)
    {
      return true;
    }

    else if (kernel_type == r2c_none || kernel_type == r2c_decrease || kernel_type == r2c_increase ||
             kernel_type == c2c_fwd_none || kernel_type == c2c_fwd_decrease || kernel_type == c2c_fwd_increase ||
             kernel_type == c2c_inv_none || kernel_type == c2c_inv_decrease || kernel_type == c2c_inv_increase ||
             kernel_type == c2r_none || kernel_type == c2r_decrease || kernel_type == c2r_increase ||
             kernel_type == xcorr_increase )
    { 
      return false;
    }
    else
    {
      std::cerr << "Function IsThreadType does not recognize the kernel type ( " << KernelName[kernel_type] << " )" << std::endl;
      exit(-1);
    }
  };

  inline bool IsR2CType(KernelType kernel_type)
  {
     if (kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
         kernel_type == r2c_none || kernel_type == r2c_decrease || kernel_type == r2c_increase)
    {
      return true;
    }   
    else return false; 

  }

  inline bool IsC2RType(KernelType kernel_type)
  {
     if (kernel_type == c2r_decomposed || kernel_type == c2r_decomposed_transposed ||
         kernel_type == c2r_none || kernel_type == c2r_decrease || kernel_type == c2r_increase)
    {
      return true;
    }  
    else return false; 
  }

  inline bool IsForwardType(KernelType kernel_type)
  {
      if (kernel_type == r2c_decomposed || kernel_type == r2c_decomposed_transposed ||
          kernel_type == r2c_none || kernel_type == r2c_decrease || kernel_type == r2c_increase ||
          kernel_type == c2c_fwd_none || kernel_type == c2c_fwd_decrease || kernel_type == c2c_fwd_increase)

    {
      return true;
    }      
    else return false; 

  }

  inline void AssertDivisibleAndFactorOf2( int full_size_transform, int number_non_zero_inputs_or_outputs)
  {
    // FIXME: This function could be named more appropriately.
    transform_size.N = full_size_transform;
    transform_size.L = number_non_zero_inputs_or_outputs;
    // FIXME: in principle, transform_size.L should equal number_non_zero_inputs_or_outputs and transform_size.P only needs to be >= and satisfy other requirements, e.g. power of two (currently.)
    transform_size.P = number_non_zero_inputs_or_outputs;

    if (transform_size.N % transform_size.P == 0) { transform_size.Q = transform_size.N / transform_size.P; }
    else { std::cerr << "Array size " << transform_size.N << " is not divisible by wanted output size " << transform_size.P << std::endl; exit(1); }

    if ( abs(fmod(log2(float(transform_size.P)), 1)) > 1e-6 ) { std::cerr << "Wanted output size " << transform_size.P << " is not a power of 2." << std::endl; exit(1); }
  }

  void GetTransformSize(KernelType kernel_type);
  void GetTransformSize_thread(KernelType kernel_type, int thread_fft_size);
  LaunchParams SetLaunchParameters(const int& ept, KernelType kernel_type, bool do_forward_transform = true);  

  inline int ReturnPaddedMemorySize(short4 & wanted_dims)
  {
    // Assumes a) SetInputDimensionsAndType has been called and is_fftw_padded is set before this call. (Currently RuntimeAssert to die if false) FIXME
    int wanted_memory = 0;

    if (is_real_valued_input)
    {
      if (wanted_dims.x % 2 == 0) { padding_jump_val = 2; wanted_memory = wanted_dims.x / 2 + 1;}
      else { padding_jump_val = 1 ; wanted_memory = (wanted_dims.x - 1) / 2 + 1;}
    
      wanted_memory *= wanted_dims.y * wanted_dims.z; // other dimensions
      wanted_memory *= 2; // room for complex
      wanted_dims.w = (wanted_dims.x + padding_jump_val) / 2; // number of complex elements in the X dimesnions after FFT.
      compute_memory_allocated = std::max(compute_memory_allocated, 2 * wanted_memory); // scaling by 2 making room for the buffer.
    }
    else
    {    
      wanted_memory = wanted_dims.x * wanted_dims.y * wanted_dims.z;
      wanted_dims.w = wanted_dims.x; // pitch is constant
      // We allocate using sizeof(ComputeType) which is either __half or float, so we need an extra factor of 2
      // Note: this must be considered when setting the address of the buffer memory based on the address of the regular memory.
      compute_memory_allocated = std::max(compute_memory_allocated, 4 * wanted_memory); 
    }
    return wanted_memory;
  }



  template<class FFT, class invFFT> void FFT_C2C_WithPadding_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);
  template<class FFT, class invFFT> void FFT_C2C_decomposed_ConjMul_C2C_t(float2* image_to_search, bool swap_real_space_quadrants);

  // 1. 
  // First call passed from a public transform function, selects block or thread and the transform precision.
  template <bool use_thread_method = false>
  void SetPrecisionAndExectutionMethod(KernelType kernel_type, bool do_forward_transform = true);

  // 2.
  // Second call, sets size of the transform kernel, selects the appropriate GPU arch
  template <class FFT_base>
  void SelectSizeAndType(KernelType kernel_type, bool do_forward_transform);

  // 3.
  // Third call, sets the input and output dimensions and type
  template <class FFT_base_arch, bool use_thread_method = false>
  void SetAndLaunchKernel(KernelType kernel_type, bool do_forward_transform);



  void PrintLaunchParameters(LaunchParams LP)
  {
    std::cout << "Launch parameters: " << std::endl;
    std::cout << "  Threads per block: ";
    PrintVectorType(LP.threadsPerBlock);
    std::cout << "  Grid dimensions: ";
    PrintVectorType(LP.gridDims);
    std::cout << "  Q: " << LP.Q << std::endl;
    std::cout << "  Twiddle in: " << LP.twiddle_in << std::endl;
    std::cout << "  shared input: " << LP.mem_offsets.shared_input << std::endl;
    std::cout << "  shared output (memlimit in r2c): " << LP.mem_offsets.shared_output << std::endl;
    std::cout << "  pixel pitch input: " << LP.mem_offsets.pixel_pitch_input << std::endl;
    std::cout << "  pixel pitch output: " << LP.mem_offsets.pixel_pitch_output << std::endl;

  };
 


}; // class Fourier Transformer

} // namespace fast_FFT



#endif
