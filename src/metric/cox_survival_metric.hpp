/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_COX_SURVIVAL_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_COX_SURVIVAL_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {

/*!
* \brief Negative partial log-likelihood metric for Cox PH models (Breslow's method).
*
* Labels encode censoring via sign: +t = event at time t, -t = censored at t.
* Lower is better.
*/
class CoxNLLMetric : public Metric {
 public:
  explicit CoxNLLMetric(const Config&) {}

  ~CoxNLLMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("survival_cox_nll");
    num_data_ = num_data;
    label_ = metadata.label();

    // Build sorted indices by ascending |label| (survival time)
    sorted_indices_.resize(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_indices_[i] = i;
    }
    std::stable_sort(sorted_indices_.begin(), sorted_indices_.end(),
                     [this](data_size_t a, data_size_t b) {
                       return std::fabs(label_[a]) < std::fabs(label_[b]);
                     });
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0;  // lower is better
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    // Breslow forward-pass to compute negative partial log-likelihood
    double max_p = score[sorted_indices_[0]];
    for (data_size_t k = 1; k < num_data_; ++k) {
      const data_size_t idx = sorted_indices_[k];
      if (score[idx] > max_p) {
        max_p = score[idx];
      }
    }

    double exp_p_sum = 0.0;
    for (data_size_t k = 0; k < num_data_; ++k) {
      exp_p_sum += std::exp(score[sorted_indices_[k]] - max_p);
    }

    double last_exp_p = 0.0;
    double last_abs_y = 0.0;
    double accumulated_sum = 0.0;
    double pll = 0.0;
    int n_events = 0;

    for (data_size_t k = 0; k < num_data_; ++k) {
      const data_size_t idx = sorted_indices_[k];
      const double p = score[idx];
      const double exp_p = std::exp(p - max_p);
      const double y = static_cast<double>(label_[idx]);
      const double abs_y = std::fabs(y);

      accumulated_sum += last_exp_p;
      if (last_abs_y < abs_y) {
        exp_p_sum -= accumulated_sum;
        accumulated_sum = 0.0;
      }

      const double safe_exp_p_sum = std::max(exp_p_sum, 1e-100);

      if (y > 0) {
        // p - log(sum exp(p_k)) = (p - max_p) - log(sum exp(p_k - max_p))
        pll += (p - max_p) - std::log(safe_exp_p_sum);
        n_events += 1;
      }

      last_abs_y = abs_y;
      last_exp_p = exp_p;
    }

    double loss = -pll / std::max(n_events, 1);
    return std::vector<double>(1, loss);
  }

 private:
  data_size_t num_data_;
  const label_t* label_;
  std::vector<data_size_t> sorted_indices_;
  std::vector<std::string> name_;
};

/*!
* \brief Harrell's concordance index metric for Cox PH models.
*
* Higher predictions = higher risk. A pair (i, j) is comparable if subject i
* had an event and T_i < |T_j|.
* Returns value in [0, 1] where 0.5 = random, 1.0 = perfect.
* Higher is better.
*/
class ConcordanceIndexMetric : public Metric {
 public:
  explicit ConcordanceIndexMetric(const Config&) {}

  ~ConcordanceIndexMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("concordance_index");
    num_data_ = num_data;
    label_ = metadata.label();
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0;  // higher is better
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    int64_t concordant = 0;
    int64_t discordant = 0;
    int64_t tied_risk = 0;

    for (data_size_t i = 0; i < num_data_; ++i) {
      const double y_i = static_cast<double>(label_[i]);
      if (y_i <= 0) {
        continue;  // i must have an event
      }
      const double t_i = y_i;
      for (data_size_t j = 0; j < num_data_; ++j) {
        if (i == j) {
          continue;
        }
        const double t_j = std::fabs(static_cast<double>(label_[j]));
        if (t_j <= t_i) {
          continue;  // j must survive longer than i
        }
        // comparable pair: i had event at t_i, j still at risk at t_i
        if (score[i] > score[j]) {
          concordant += 1;
        } else if (score[i] < score[j]) {
          discordant += 1;
        } else {
          tied_risk += 1;
        }
      }
    }

    const int64_t total = concordant + discordant + tied_risk;
    double c_index = 0.5;
    if (total > 0) {
      c_index = (static_cast<double>(concordant) + 0.5 * static_cast<double>(tied_risk))
                / static_cast<double>(total);
    }
    return std::vector<double>(1, c_index);
  }

 private:
  data_size_t num_data_;
  const label_t* label_;
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_METRIC_COX_SURVIVAL_METRIC_HPP_
