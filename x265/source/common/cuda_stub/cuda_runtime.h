/*****************************************************************************
 * cuda_runtime.h: minimal stub for the device-only kernel build
 *****************************************************************************
 * gpu_kernels.cu is compiled with `nvcc -fatbin` (device code only) and the
 * nvcc driver includes "cuda_runtime.h" implicitly.  The real header pulls in
 * the host runtime API and, through it, the C++ standard library, which the
 * kernel build does not need and which the latest MSVC standard library
 * refuses to be parsed by older CUDA front ends ("expected CUDA 13.2 or
 * newer").  This stub is placed in front of the CUDA include path so that the
 * implicit include resolves here instead.
 *
 * The kernels need the __global__ / __device__ / __forceinline__ / __restrict__
 * macros (crt/host_defines.h) and the blockIdx / blockDim / threadIdx
 * variables (device_launch_parameters.h); those two are self-contained and
 * free of host or standard library declarations.  min()/max() are provided by
 * gpu_kernels.cu itself.
 *****************************************************************************/

#ifndef X265_GPU_CUDA_RUNTIME_STUB_H
#define X265_GPU_CUDA_RUNTIME_STUB_H

/* Silences the "crt/host_defines.h is an internal header" notice. */
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#include <device_launch_parameters.h>

#endif
