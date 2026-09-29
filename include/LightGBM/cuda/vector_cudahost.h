/*!
 * Copyright (c) 2020-2021 IBM Corporation, Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_

#include <LightGBM/utils/common.h>

#include <cstdint>
#include <new>

#ifdef USE_CUDA
#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // USE_ROCM
#include <LightGBM/cuda/cuda_utils.hu>
#endif  // USE_CUDA
#include <stdio.h>

enum LGBM_Device {
  lgbm_device_cpu,
  lgbm_device_gpu,
  lgbm_device_cuda
};

enum Use_Learner {
  use_cpu_learner,
  use_gpu_learner,
  use_cuda_learner
};

namespace LightGBM {

class LGBM_config_ {
 public:
  static int current_device;  // Default: lgbm_device_cpu
  static int current_learner;  // Default: use_cpu_learner
};


template <class T>
struct CHAllocator {
 private:
  struct AllocationHeader {
    void* base;
    bool cuda_host_alloc;
  };

  static std::size_t Alignment() {
    std::size_t alignment = 16;
    if (alignment < alignof(T)) {
      alignment = alignof(T);
    }
    if (alignment < alignof(AllocationHeader)) {
      alignment = alignof(AllocationHeader);
    }
    return alignment;
  }

 public:
  typedef T value_type;
  CHAllocator() {}
  template <class U> CHAllocator(const CHAllocator<U>& other);
  T* allocate(std::size_t n) {
    if (n == 0) return NULL;
    n = SIZE_ALIGNED(n);
    const std::size_t alignment = Alignment();
    const std::size_t allocation_size = n * sizeof(T) + sizeof(AllocationHeader) + alignment - 1;
    void* base = nullptr;
    bool cuda_host_alloc = false;
    #ifdef USE_CUDA
      if (LGBM_config_::current_device == lgbm_device_cuda) {
        const cudaError_t ret = cudaHostAlloc(&base, allocation_size, cudaHostAllocPortable);
        if (ret != cudaSuccess) {
          Log::Warning("Defaulting to malloc in CHAllocator!!!");
          base = _mm_malloc(allocation_size, alignment);
        } else {
          cuda_host_alloc = true;
        }
      } else {
        base = _mm_malloc(allocation_size, alignment);
      }
    #else
      base = _mm_malloc(allocation_size, alignment);
    #endif
    if (base == nullptr) {
      return nullptr;
    }
    const std::uintptr_t first_address = reinterpret_cast<std::uintptr_t>(base) + sizeof(AllocationHeader);
    const std::uintptr_t aligned_address = (first_address + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    auto* header = reinterpret_cast<AllocationHeader*>(aligned_address - sizeof(AllocationHeader));
    new (header) AllocationHeader{base, cuda_host_alloc};
    return reinterpret_cast<T*>(aligned_address);
  }

  void deallocate(T* p, std::size_t n) {
    (void)n;  // UNUSED
    if (p == NULL) return;
    auto* header = reinterpret_cast<AllocationHeader*>(reinterpret_cast<std::uintptr_t>(p) - sizeof(AllocationHeader));
    void* base = header->base;
    const bool cuda_host_alloc = header->cuda_host_alloc;
    header->~AllocationHeader();
    #ifdef USE_CUDA
      if (cuda_host_alloc) {
        CUDASUCCESS_OR_FATAL(cudaFreeHost(base));
      } else {
        _mm_free(base);
      }
    #else
      _mm_free(base);
    #endif
  }
};
template <class T, class U>
bool operator==(const CHAllocator<T>&, const CHAllocator<U>&);
template <class T, class U>
bool operator!=(const CHAllocator<T>&, const CHAllocator<U>&);

}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_VECTOR_CUDAHOST_H_
