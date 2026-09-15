/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_multiclass_metric.hpp"

#include <algorithm>
#include <cstring>

namespace LightGBM {

CUDAMultiSoftmaxLoglossMetric::CUDAMultiSoftmaxLoglossMetric(const Config& config):
  CUDAMetricInterface<MultiSoftmaxLoglossMetric>(config),
  num_class_(config.num_class),
  sigmoid_(config.sigmoid) {}

void CUDAMultiSoftmaxLoglossMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<MultiSoftmaxLoglossMetric>::Init(metadata, num_data);
  num_data_ = num_data;
  sum_weights_ = static_cast<double>(num_data_);
  if (metadata.weights() != nullptr) {
    sum_weights_ = 0.0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      sum_weights_ += metadata.weights()[i];
    }
  }
  const int num_blocks = static_cast<int>((num_data_ + 255) / 256);
  reduce_block_buffer_.Resize(static_cast<size_t>(num_blocks));
  const int num_reduce_blocks = (num_blocks + 1024 - 1) / 1024;
  reduce_block_buffer_inner_.Resize(static_cast<size_t>(std::max(1, num_reduce_blocks)));
}

std::vector<double> CUDAMultiSoftmaxLoglossMetric::Eval(
    const double* score, const ObjectiveFunction* objective) const {
  // mode 0: scores are already probabilities (no objective), mode 1: softmax
  // logits, and mode 2: one-vs-all logits.  The latter keeps this metric
  // compatible with multiclassova while avoiding a temporary probability
  // matrix on the device.
  int mode = 0;
  if (objective != nullptr) {
    mode = std::strcmp(objective->GetName(), "multiclassova") == 0 ? 2 : 1;
  }
  double sum_loss = 0.0;
  LaunchEvalKernel(score, mode, sigmoid_, &sum_loss);
  return std::vector<double>{sum_loss / sum_weights_};
}

}  // namespace LightGBM

#endif  // USE_CUDA
