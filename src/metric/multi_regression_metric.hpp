/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_MULTI_REGRESSION_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_MULTI_REGRESSION_METRIC_HPP_

#include <LightGBM/metric.h>

#include <cmath>
#include <string>
#include <vector>

namespace LightGBM {
/*!
* \brief L2 / RMSE with num_target > 1, computed over all samples and targets (same as XGBoost).
*        Labels and scores are column-major: index = num_data * target + row.
*/
template<bool kRoot>
class MultiRegressionL2Metric: public Metric {
 public:
  explicit MultiRegressionL2Metric(const Config& config) : num_target_(config.num_target) {}

  const std::vector<std::string>& GetName() const override { return name_; }

  double factor_to_bigger_better() const override { return -1.0f; }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(kRoot ? "rmse" : "l2");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    if (metadata.num_label_columns() != num_target_) {
      Log::Fatal("num_target is %d, but label has %d columns", num_target_, metadata.num_label_columns());
    }
    sum_weights_ = static_cast<double>(num_data_);
    if (weights_ != nullptr) {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  // scores are raw: multi-target regression has an identity output transformation
  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    const int64_t total = static_cast<int64_t>(num_data_) * num_target_;
    double sum_loss = 0.0f;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
    for (int64_t idx = 0; idx < total; ++idx) {
      const double diff = score[idx] - label_[idx];
      sum_loss += (weights_ == nullptr) ? diff * diff : diff * diff * weights_[idx % num_data_];
    }
    const double loss = sum_loss / (sum_weights_ * num_target_);
    return std::vector<double>(1, kRoot ? std::sqrt(loss) : loss);
  }

 private:
  const int num_target_;
  data_size_t num_data_;
  const label_t* label_;
  const label_t* weights_;
  double sum_weights_;
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_METRIC_MULTI_REGRESSION_METRIC_HPP_
