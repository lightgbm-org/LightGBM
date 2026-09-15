/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.hu>

#include "../multiclass_metric.hpp"

#include <vector>

namespace LightGBM {

class CUDAMultiSoftmaxLoglossMetric : public CUDAMetricInterface<MultiSoftmaxLoglossMetric> {
 public:
  explicit CUDAMultiSoftmaxLoglossMetric(const Config& config);

  ~CUDAMultiSoftmaxLoglossMetric() override = default;

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 private:
  void LaunchEvalKernel(const double* score, int mode, double sigmoid,
                        double* sum_loss) const;

  mutable CUDAVector<double> reduce_block_buffer_;
  mutable CUDAVector<double> reduce_block_buffer_inner_;
  data_size_t num_data_ = 0;
  int num_class_ = 0;
  double sigmoid_ = 1.0;
  double sum_weights_ = 0.0;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
