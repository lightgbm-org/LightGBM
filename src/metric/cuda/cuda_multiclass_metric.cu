/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include "cuda_multiclass_metric.hpp"

#include <cmath>

namespace LightGBM {

namespace {

constexpr int kMetricBlockSize = 256;
constexpr int kProbabilityMode = 0;
constexpr int kSoftmaxMode = 1;
constexpr int kOVAMode = 2;

template <bool USE_WEIGHTS>
__global__ void MultiLoglossKernel(const data_size_t num_data, const int num_class,
                                   const label_t* labels, const label_t* weights,
                                   const double* scores, const int mode,
                                   const double sigmoid, double* reduce_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  double loss = 0.0;
  if (data_index < num_data) {
    const int label = static_cast<int>(labels[data_index]);
    if (mode == kProbabilityMode) {
      const double probability = scores[static_cast<size_t>(label) * num_data + data_index];
      loss = probability > kEpsilon ? -log(probability) : -log(kEpsilon);
    } else if (mode == kOVAMode) {
      const double z = sigmoid * scores[static_cast<size_t>(label) * num_data + data_index];
      // Stable log(1 + exp(-z)).
      loss = z >= 0.0 ? log1p(exp(-z)) : -z + log1p(exp(z));
      loss = loss < -log(kEpsilon) ? loss : -log(kEpsilon);
    } else {
      double max_score = scores[data_index];
      for (int class_id = 1; class_id < num_class; ++class_id) {
        max_score = max(max_score, scores[static_cast<size_t>(class_id) * num_data + data_index]);
      }
      double sum_exp = 0.0;
      for (int class_id = 0; class_id < num_class; ++class_id) {
        sum_exp += exp(scores[static_cast<size_t>(class_id) * num_data + data_index] - max_score);
      }
      const double log_probability = scores[static_cast<size_t>(label) * num_data + data_index] -
        max_score - log(sum_exp);
      loss = log_probability > log(kEpsilon) ? -log_probability : -log(kEpsilon);
    }
    if (USE_WEIGHTS) {
      loss *= weights[data_index];
    }
  }
  const double block_loss = ShuffleReduceSum<double>(loss, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    reduce_buffer[blockIdx.x] = block_loss;
  }
}

}  // namespace

void CUDAMultiSoftmaxLoglossMetric::LaunchEvalKernel(const double* score, const int mode,
                                                     const double sigmoid,
                                                     double* sum_loss) const {
  const int num_blocks = static_cast<int>((num_data_ + kMetricBlockSize - 1) / kMetricBlockSize);
  if (this->cuda_weights_ == nullptr) {
    MultiLoglossKernel<false><<<num_blocks, kMetricBlockSize>>>(
      num_data_, num_class_, this->cuda_labels_, nullptr, score, mode, sigmoid,
      reduce_block_buffer_.RawData());
  } else {
    MultiLoglossKernel<true><<<num_blocks, kMetricBlockSize>>>(
      num_data_, num_class_, this->cuda_labels_, this->cuda_weights_, score, mode, sigmoid,
      reduce_block_buffer_.RawData());
  }
  ShuffleReduceSumGlobal<double, double>(reduce_block_buffer_.RawData(),
                                         static_cast<size_t>(num_blocks),
                                         reduce_block_buffer_inner_.RawData());
  CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer_inner_.RawData(), 1, __FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
