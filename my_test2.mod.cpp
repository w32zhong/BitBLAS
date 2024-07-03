// gcc my_test2.mod.cpp -I 3rdparty/tvm/include/ -I 3rdparty/tvm/3rdparty/dlpack/include/
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t main(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
typedef uint16_t half;
TVM_DLL int32_t main(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t A_handle_code = arg_type_ids[0];
  int32_t B_handle_code = arg_type_ids[1];
  int32_t B_decode_handle_code = arg_type_ids[2];
  void* A_handle = (((TVMValue*)args)[0].v_handle);
  void* B_handle = (((TVMValue*)args)[1].v_handle);
  void* B_decode_handle = (((TVMValue*)args)[2].v_handle);
  void* main_A_handle_shape = (((DLTensor*)A_handle)[0].shape);
  void* main_A_handle_strides = (((DLTensor*)A_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)A_handle)[0].device.device_id);
  void* A = (((DLTensor*)A_handle)[0].data);
  void* main_B_handle_shape = (((DLTensor*)B_handle)[0].shape);
  void* main_B_handle_strides = (((DLTensor*)B_handle)[0].strides);
  void* B = (((DLTensor*)B_handle)[0].data);
  void* main_B_decode_handle_shape = (((DLTensor*)B_decode_handle)[0].shape);
  void* main_B_decode_handle_strides = (((DLTensor*)B_decode_handle)[0].strides);
  void* B_decode = (((DLTensor*)B_decode_handle)[0].data);
  if (!(main_A_handle_strides == NULL)) {
  }
  if (!(main_B_handle_strides == NULL)) {
  }
  if (!(main_B_decode_handle_strides == NULL)) {
  }
  for (int32_t n = 0; n < 1024; ++n) {
    for (int32_t k = 0; k < 1024; ++k) {
      ((half*)B_decode)[((n * 1024) + k)] = (((half)((((uint32_t)((int8_t*)B)[((n * 256) + (k >> 2))]) >> (((uint32_t)(k & 3)) * (uint32_t)2)) & (uint32_t)3)) - (half)2.000000e+00f);
    }
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return main(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
