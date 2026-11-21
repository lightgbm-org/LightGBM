/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {
/*!
* \brief L2 regression with num_target > 1, one tree per target per iteration.
*        Labels and scores are column-major: index = num_data * target + row.
*        Per target, this is identical to RegressionL2loss.
*/
class MultiRegressionL2loss: public ObjectiveFunction {
 public:
  explicit MultiRegressionL2loss(const Config& config)
      : num_target_(config.num_target), deterministic_(config.deterministic) {
  }

  explicit MultiRegressionL2loss(const std::vector<std::string>& strs)
      : num_target_(ParseNumTarget(strs)), deterministic_(false) {
  }

  ~MultiRegressionL2loss() {}

  /*! \brief Number of targets stored in a model string, 1 if absent */
  static int ParseNumTarget(const std::vector<std::string>& strs) {
    int num_target = 1;
    for (auto str : strs) {
      auto tokens = Common::Split(str.c_str(), ':');
      if (tokens.size() == 2 && tokens[0] == std::string("num_target")) {
        Common::Atoi(tokens[1].c_str(), &num_target);
      }
    }
    return num_target;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    if (metadata.num_label_columns() != num_target_) {
      Log::Fatal("num_target is %d, but label has %d columns", num_target_, metadata.num_label_columns());
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    const int64_t total = static_cast<int64_t>(num_data_) * num_target_;
    if (weights_ == nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int64_t idx = 0; idx < total; ++idx) {
        gradients[idx] = static_cast<score_t>(score[idx] - label_[idx]);
        hessians[idx] = 1.0f;
      }
    } else {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int64_t idx = 0; idx < total; ++idx) {
        const label_t w = weights_[idx % num_data_];
        gradients[idx] = static_cast<score_t>(static_cast<score_t>((score[idx] - label_[idx])) * w);
        hessians[idx] = static_cast<score_t>(w);
      }
    }
  }

  const char* GetName() const override {
    return "regression";
  }

  void ConvertOutput(const double* input, double* output) const override {
    for (int k = 0; k < num_target_; ++k) {
      output[k] = input[k];
    }
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName() << " num_target:" << num_target_;
    return str_buf.str();
  }

  bool IsConstantHessian() const override { return weights_ == nullptr; }

  int NumModelPerIteration() const override { return num_target_; }

  int NumPredictOneRow() const override { return num_target_; }

  double BoostFromScore(int class_id) const override {
    const label_t* label = label_ + static_cast<size_t>(num_data_) * class_id;
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml, sumw) if (!deterministic_)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += static_cast<double>(label[i]) * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml) if (!deterministic_)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label[i];
      }
    }
    return suml / sumw;
  }

 private:
  const int num_target_;
  const bool deterministic_;
  data_size_t num_data_;
  const label_t* label_;
  const label_t* weights_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_
