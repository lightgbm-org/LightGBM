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
* \brief L2 / RMSE for multi-target regression, computed per target then averaged over targets
*        (same as scikit-learn's multioutput="uniform_average").
*        Labels and scores are column-major: index = num_data * target + row.
*/
template<bool kRoot>
class MultiRegressionL2Metric: public Metric {
 public:
  explicit MultiRegressionL2Metric(const Config& config) : num_target_(config.num_class) {}

  const std::vector<std::string>& GetName() const override { return name_; }

  double factor_to_bigger_better() const override { return -1.0f; }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(kRoot ? "multi_rmse" : "multi_l2");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    if (metadata.num_label_columns() != num_target_) {
      Log::Fatal("Metric %s expects %d label columns (num_class), but found %d",
                 name_[0].c_str(), num_target_, metadata.num_label_columns());
    }
    sum_weights_ = static_cast<double>(num_data_);
    if (weights_ != nullptr) {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  // scores are raw: multi_regression has an identity output transformation
  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    double sum_over_targets = 0.0f;
    for (int k = 0; k < num_target_; ++k) {
      const size_t offset = static_cast<size_t>(num_data_) * k;
      double sum_loss = 0.0f;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[offset + i] - label_[offset + i];
        sum_loss += (weights_ == nullptr) ? diff * diff : diff * diff * weights_[i];
      }
      const double loss = sum_loss / sum_weights_;
      sum_over_targets += kRoot ? std::sqrt(loss) : loss;
    }
    return std::vector<double>(1, sum_over_targets / num_target_);
  }

 private:
  int num_target_;
  data_size_t num_data_;
  const label_t* label_;
  const label_t* weights_;
  double sum_weights_;
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_METRIC_MULTI_REGRESSION_METRIC_HPP_
