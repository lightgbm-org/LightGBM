/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_COX_SURVIVAL_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_COX_SURVIVAL_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {
/*!
* \brief Objective function for Cox proportional hazards survival analysis.
*
* Implements the Cox partial log-likelihood with Breslow's method for ties.
* Labels encode censoring via sign: +t = event at time t, -t = censored at t.
*/
class CoxPHLoss : public ObjectiveFunction {
 public:
  explicit CoxPHLoss(const Config&) {}

  explicit CoxPHLoss(const std::vector<std::string>&) {}

  ~CoxPHLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    // Build sorted indices by ascending |label| (survival/censoring time)
    sorted_indices_.resize(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_indices_[i] = i;
    }
    std::stable_sort(sorted_indices_.begin(), sorted_indices_.end(),
                     [this](data_size_t a, data_size_t b) {
                       return std::fabs(label_[a]) < std::fabs(label_[b]);
                     });
  }

  void GetGradients(const double* score,
                    score_t* gradients, score_t* hessians) const override {
    // Breslow's method for Cox PH gradients.
    // Iterate through data in sorted time order.

    // find max score for numerical stability
    double max_p = score[0];
    for (data_size_t k = 1; k < num_data_; ++k) {
      if (score[k] > max_p) {
        max_p = score[k];
      }
    }

    // sum of exp(score - max_p) over all samples
    double exp_p_sum = 0.0;
    for (data_size_t k = 0; k < num_data_; ++k) {
      exp_p_sum += std::exp(score[k] - max_p);
    }

    // forward pass in sorted order
    double r_k = 0.0;
    double s_k = 0.0;
    double last_exp_p = 0.0;
    double last_abs_y = 0.0;
    double accumulated_sum = 0.0;

    for (data_size_t k = 0; k < num_data_; ++k) {
      const data_size_t idx = sorted_indices_[k];
      const double p = score[idx];
      const double exp_p = std::exp(p - max_p);
      const double y = static_cast<double>(label_[idx]);
      const double abs_y = std::fabs(y);

      // Only update the denominator after we move forward in time
      // This is Breslow's method for ties
      accumulated_sum += last_exp_p;
      if (last_abs_y < abs_y) {
        exp_p_sum -= accumulated_sum;
        accumulated_sum = 0.0;
      }

      if (y > 0) {
        r_k += 1.0 / exp_p_sum;
        s_k += 1.0 / (exp_p_sum * exp_p_sum);
      }

      double g = exp_p * r_k - static_cast<double>(y > 0);
      double h = exp_p * r_k - exp_p * exp_p * s_k;

      if (weights_ != nullptr) {
        g *= weights_[idx];
        h *= weights_[idx];
      }

      gradients[idx] = static_cast<score_t>(g);
      hessians[idx] = static_cast<score_t>(h);

      last_abs_y = abs_y;
      last_exp_p = exp_p;
    }
  }

  const char* GetName() const override {
    return "survival_cox";
  }

  std::string ToString() const override {
    return std::string("survival_cox");
  }

  bool NeedAccuratePrediction() const override { return false; }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label (signed survival times) */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Indices sorted by ascending |label| (survival time) */
  std::vector<data_size_t> sorted_indices_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_OBJECTIVE_COX_SURVIVAL_OBJECTIVE_HPP_
