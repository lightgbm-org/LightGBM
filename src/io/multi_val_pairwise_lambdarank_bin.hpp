/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_

#include "multi_val_dense_bin.hpp"

#include <cstring>

namespace LightGBM {

template <typename BIN_TYPE, template<typename> class MULTI_VAL_BIN_TYPE>
class MultiValPairwiseLambdarankBin : public MULTI_VAL_BIN_TYPE<BIN_TYPE> {
 public:
  MultiValPairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature,
                                const std::vector<uint32_t>& offsets)
      : MULTI_VAL_BIN_TYPE<BIN_TYPE>(num_data, num_bin, num_feature, offsets) {
    this->num_bin_ = num_bin;
    Log::Warning("num_bin = %d", num_bin);
  }

 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map_;
};

template <typename BIN_TYPE>
class MultiValDensePairwiseLambdarankBin
    : public MultiValPairwiseLambdarankBin<BIN_TYPE, MultiValDenseBin> {
 public:
  MultiValDensePairwiseLambdarankBin(
      data_size_t num_data, int num_bin, int num_feature,
      const std::vector<uint32_t>& offsets,
      const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map,
      const std::vector<const BinMapper*> diff_feature_bin_mappers,
      const std::vector<const BinMapper*> original_feature_bin_mappers,
      const bool use_pairwise_bin_lookup,
      const std::vector<std::vector<float>>* raw_data,
      const std::vector<uint32_t>& all_offsets,
      const std::vector<int>& diff_feature_to_original_feature_slot,
      const std::vector<int>& diff_feature_to_raw_feature_index,
      const std::string differential_feature_bin_mode)
      : MultiValPairwiseLambdarankBin<BIN_TYPE, LightGBM::MultiValDenseBin>(
            num_data, num_bin, num_feature, offsets),
        diff_feature_to_original_feature_slot_(diff_feature_to_original_feature_slot),
        pointwise_histogram_buffers_(OMP_NUM_THREADS()),
        all_hist_offsets_(all_offsets),
        use_fast_pairwise_bin_lookup_(use_pairwise_bin_lookup),
        differential_feature_bin_mode_(differential_feature_bin_mode) {
    this->paired_ranking_item_global_index_map_ =
        paired_ranking_item_global_index_map;
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_original_feature_slot.size());
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_raw_feature_index.size());
    num_diff_features_ = static_cast<int>(diff_feature_bin_mappers.size());
    diff_feature_bin_mappers_.reserve(num_diff_features_);
    diff_feature_hist_offsets_.reserve(num_diff_features_);
    diff_feature_bin_value_offsets_.reserve(num_diff_features_);
    original_feature_bin_value_offsets_.reserve(num_diff_features_);
    diff_feature_raw_data_ptrs_.reserve(num_diff_features_);
    original_feature_most_freq_bins_.reserve(num_diff_features_);
    original_feature_num_bins_.reserve(num_diff_features_);
    active_diff_feature_slots_.reserve(num_diff_features_);
    active_diff_feature_hist_offsets_.reserve(num_diff_features_);
    active_diff_feature_bin_value_offsets_.reserve(num_diff_features_);
    active_diff_feature_bin_mappers_.reserve(num_diff_features_);
    for (int j = 0; j < num_diff_features_; ++j) {
      diff_feature_bin_mappers_.emplace_back(diff_feature_bin_mappers[j]);
      diff_feature_hist_offsets_.push_back(all_offsets[2 * this->num_feature_ + j]);
      diff_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(diff_feature_bin_mappers_[j]->GetMostFreqBin() == 0));
      original_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(original_feature_bin_mappers[j]->GetMostFreqBin() == 0));
      original_feature_most_freq_bins_.push_back(original_feature_bin_mappers[j]->GetMostFreqBin());
      original_feature_num_bins_.push_back(original_feature_bin_mappers[j]->num_bin());
      const float* raw_data_ptr = nullptr;
      if (raw_data != nullptr && !raw_data->empty()) {
        const int raw_feature_index = diff_feature_to_raw_feature_index[j];
        CHECK_GE(raw_feature_index, 0);
        CHECK_LT(raw_feature_index, static_cast<int>(raw_data->size()));
        raw_data_ptr = (*raw_data)[raw_feature_index].data();
      }
      diff_feature_raw_data_ptrs_.push_back(raw_data_ptr);
      if (differential_feature_bin_mode_ == "lookup" || raw_data_ptr != nullptr) {
        active_diff_feature_slots_.push_back(j);
        active_diff_feature_hist_offsets_.push_back(diff_feature_hist_offsets_[j]);
        active_diff_feature_bin_value_offsets_.push_back(diff_feature_bin_value_offsets_[j]);
        active_diff_feature_bin_mappers_.push_back(diff_feature_bin_mappers_[j].get());
        active_diff_feature_to_original_feature_slot_.push_back(diff_feature_to_original_feature_slot_[j]);
        use_fast_pairwise_bin_lookup_ &= differential_feature_bin_mode_ == "lookup"
            ? diff_feature_bin_mappers_[j]->HasPairwiseBinLookup()
            : diff_feature_bin_mappers_[j]->HasPairwiseBinRanges();
      }
    }
    InitDiffFeatureRawValueBuffer(num_data);
  }

  void FinishLoad() override {
    if (differential_feature_bin_mode_ != "lookup") {
      return;
    }
    // Decode packed pointwise bins once. Store small feature tiles together so
    // each document access supplies all of the histograms in a tile.
    const int num_tiles = (num_active_diff_features_ + kLookupFeatureTileSize - 1) / kLookupFeatureTileSize;
    lookup_feature_bins_.resize(static_cast<size_t>(this->num_data_) * num_tiles * kLookupFeatureTileSize);
    #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
    for (int tile = 0; tile < num_tiles; ++tile) {
      const int first_feature = tile * kLookupFeatureTileSize;
      const int tile_size = std::min(kLookupFeatureTileSize, num_active_diff_features_ - first_feature);
      auto* tile_bins = lookup_feature_bins_.data() + static_cast<size_t>(first_feature) * this->num_data_;
      for (data_size_t row = 0; row < this->num_data_; ++row) {
        for (int j = 0; j < tile_size; ++j) {
          const int feature = first_feature + j;
          const int slot = active_diff_feature_to_original_feature_slot_[feature];
          const uint32_t offset = original_feature_bin_value_offsets_[feature];
          const uint32_t most_freq_bin = original_feature_most_freq_bins_[feature];
          const uint32_t bin = this->data_[this->RowPtr(row) + slot];
          tile_bins[static_cast<size_t>(row) * kLookupFeatureTileSize + j] =
              static_cast<BIN_TYPE>(bin == 0 ? most_freq_bin : bin - offset);
        }
      }
    }
  }

  bool ConstructHistogramsByFeature(const data_size_t* data_indices, data_size_t num_data,
      data_size_t num_total_data, int num_blocks, data_size_t block_size, const score_t* gradients, const score_t* hessians,
      bool ordered, int num_threads, hist_t* out, const int8_t* is_feature_used = nullptr,
      HistogramBinMask* const* bin_masks = nullptr) const override {
    if (differential_feature_bin_mode_ != "lookup") {
      return false;
    }
    num_total_pairs_ = num_total_data;
    if (data_indices != nullptr) {
      if (ordered) {
        ConstructLookupHistograms<true, true>(data_indices, num_data, num_blocks,
            block_size, gradients, hessians, num_threads, out, is_feature_used, bin_masks);
      } else {
        ConstructLookupHistograms<true, false>(data_indices, num_data, num_blocks,
            block_size, gradients, hessians, num_threads, out, is_feature_used, bin_masks);
      }
    } else {
      ConstructLookupHistograms<false, false>(nullptr, num_data, num_blocks,
          block_size, gradients, hessians, num_threads, out, is_feature_used, bin_masks);
    }
    return true;
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* hessians, hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                               gradients, hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(nullptr, start, end, gradients,
                                                 hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  template <bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* gradients,
                               const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    const auto base_offset = this->offsets_.back();
    const data_size_t num_original_data = this->num_data_;
    const int tid = omp_get_thread_num();
    CHECK_LT(tid, static_cast<int>(pointwise_histogram_buffers_.size()));
    PointwiseHistogramBuffer& pointwise_buffer = pointwise_histogram_buffers_[tid];
    pointwise_buffer.Reset(num_original_data, static_cast<size_t>(end - start));
    const auto& first_contributions = pointwise_buffer.first_contributions;
    const auto& second_contributions = pointwise_buffer.second_contributions;
    std::vector<data_size_t>& first_touched = pointwise_buffer.first_touched;
    std::vector<data_size_t>& second_touched = pointwise_buffer.second_touched;
    const uint32_t stamp = pointwise_buffer.stamp;

    if (differential_feature_bin_mode_ == "lookup") {
      for (data_size_t row = start; row < end; ++row) {
        const auto idx = USE_INDICES ? data_indices[row] : row;
        const auto pair = this->paired_ranking_item_global_index_map_[idx];
        const score_t gradient = ORDERED ? gradients[row] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[row] : hessians[idx];
        pointwise_buffer.AddFirst(pair.first, gradient, hessian, stamp);
        pointwise_buffer.AddSecond(pair.second, gradient, hessian, stamp);
      }
      for (int j = 0; j < num_active_diff_features_; ++j) {
        const auto* column = lookup_feature_bins_.data() +
            static_cast<size_t>(j / kLookupFeatureTileSize) * kLookupFeatureTileSize * this->num_data_ +
            j % kLookupFeatureTileSize;
        const uint32_t num_bins = original_feature_num_bins_[j];
        auto* histogram = out + 2 * (active_diff_feature_hist_offsets_[j] +
                                     active_diff_feature_bin_value_offsets_[j]);
        for (data_size_t row = start; row < end; ++row) {
          const auto idx = USE_INDICES ? data_indices[row] : row;
          const auto pair = this->paired_ranking_item_global_index_map_[idx];
          const uint32_t bin = static_cast<uint32_t>(column[static_cast<size_t>(pair.first) * kLookupFeatureTileSize]) *
              num_bins + column[static_cast<size_t>(pair.second) * kLookupFeatureTileSize];
          histogram[2 * bin] += ORDERED ? gradients[row] : gradients[idx];
          histogram[2 * bin + 1] += ORDERED ? hessians[row] : hessians[idx];
        }
      }
    } else if (!active_diff_feature_slots_.empty() && use_fast_pairwise_bin_lookup_) {
      const float* diff_feature_raw_values = diff_feature_raw_values_.data();
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        if (USE_PREFETCH) {
          const data_size_t pf_offset = 32 / sizeof(BIN_TYPE);
          const data_size_t pf_i = i + pf_offset;
          if (pf_i < end) {
            const auto pf_idx = USE_INDICES ? data_indices[pf_i] : pf_i;
            const data_size_t pf_first_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].first;
            const data_size_t pf_second_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].second;
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_first_idx));
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_second_idx));
            if (differential_feature_bin_mode_ != "lookup") {
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_first_idx) * num_active_diff_features_);
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_second_idx) * num_active_diff_features_);
            }
            if (!ORDERED) {
              PREFETCH_T0(gradients + pf_idx);
              PREFETCH_T0(hessians + pf_idx);
            }
          }
        }
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];

        pointwise_buffer.AddFirst(first_idx, gradient, hessian, stamp);
        pointwise_buffer.AddSecond(second_idx, gradient, hessian, stamp);

        if (differential_feature_bin_mode_ == std::string("ternary")) {

          const float* first_row_diff_feature_raw_values =
              diff_feature_raw_values +
              static_cast<size_t>(first_idx) * num_active_diff_features_;
          const float* second_row_diff_feature_raw_values =
              diff_feature_raw_values +
              static_cast<size_t>(second_idx) * num_active_diff_features_;
          for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
            const float diff_value =
                first_row_diff_feature_raw_values[active_j] -
                second_row_diff_feature_raw_values[active_j];

            const uint32_t diff_bin =
                static_cast<uint32_t>(diff_value > static_cast<float>(-kEpsilon)) +
                static_cast<uint32_t>(diff_value > static_cast<float>(kEpsilon));
            // The original row-wise bins already exclude feature-group offsets.
            // Differential features still need the single-feature-group packing fix below.
            const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
            const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        } else if (differential_feature_bin_mode_ == std::string("exact")) {
          const BIN_TYPE* first_data_ptr = this->data_.data() + this->RowPtr(first_idx);
          const BIN_TYPE* second_data_ptr = this->data_.data() + this->RowPtr(second_idx);

          const float* first_row_diff_feature_raw_values =
              diff_feature_raw_values +
              static_cast<size_t>(first_idx) * num_active_diff_features_;
          const float* second_row_diff_feature_raw_values =
              diff_feature_raw_values +
              static_cast<size_t>(second_idx) * num_active_diff_features_;
          for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
            const int original_feature_slot = active_diff_feature_to_original_feature_slot_[active_j];
            uint32_t first_bin = static_cast<uint32_t>(first_data_ptr[original_feature_slot]);
            uint32_t second_bin = static_cast<uint32_t>(second_data_ptr[original_feature_slot]);
            const float diff_value =
                first_row_diff_feature_raw_values[active_j] -
                second_row_diff_feature_raw_values[active_j];
            if (first_bin >= original_feature_bin_value_offsets_[active_j]) {
              first_bin -= original_feature_bin_value_offsets_[active_j];
            } else {
              CHECK_EQ(first_bin, 0);
              first_bin = original_feature_most_freq_bins_[active_j];
            }
            if (second_bin >= original_feature_bin_value_offsets_[active_j]) {
              second_bin -= original_feature_bin_value_offsets_[active_j];
            } else {
              CHECK_EQ(second_bin, 0);
              second_bin = original_feature_most_freq_bins_[active_j];
            }

            const uint32_t diff_bin =
                active_diff_feature_bin_mappers_[active_j]->ValueToBinWithPairwiseRangeUnchecked(
                    diff_value, first_bin, second_bin, original_feature_slot);
            // The original row-wise bins already exclude feature-group offsets.
            // Differential features still need the single-feature-group packing fix below.
            const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
            const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        } else {
          const BIN_TYPE* first_data_ptr = this->data_.data() + this->RowPtr(first_idx);
          const BIN_TYPE* second_data_ptr = this->data_.data() + this->RowPtr(second_idx);

          for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
            const int original_feature_slot = active_diff_feature_to_original_feature_slot_[active_j];
            uint32_t first_bin = static_cast<uint32_t>(first_data_ptr[original_feature_slot]);
            uint32_t second_bin = static_cast<uint32_t>(second_data_ptr[original_feature_slot]);

            if (first_bin >= original_feature_bin_value_offsets_[active_j]) {
              first_bin -= original_feature_bin_value_offsets_[active_j];
            } else {
              CHECK_EQ(first_bin, 0);
              first_bin = original_feature_most_freq_bins_[active_j];
            }
            if (second_bin >= original_feature_bin_value_offsets_[active_j]) {
              second_bin -= original_feature_bin_value_offsets_[active_j];
            } else {
              CHECK_EQ(second_bin, 0);
              second_bin = original_feature_most_freq_bins_[active_j];
            }

            const uint32_t diff_bin =
                active_diff_feature_bin_mappers_[active_j]->PairwiseBinToBin(first_bin, second_bin);
            // The original row-wise bins already exclude feature-group offsets.
            // Differential features still need the single-feature-group packing fix below.
            const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
            const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
      }
    } else {
      const float* diff_feature_raw_values = diff_feature_raw_values_.data();
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        if (USE_PREFETCH) {
          const data_size_t pf_offset = 32 / sizeof(BIN_TYPE);
          const data_size_t pf_i = i + pf_offset;
          if (pf_i < end) {
            const auto pf_idx = USE_INDICES ? data_indices[pf_i] : pf_i;
            const data_size_t pf_first_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].first;
            const data_size_t pf_second_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].second;
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_first_idx));
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_second_idx));
            if (!active_diff_feature_slots_.empty()) {
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_first_idx) * num_active_diff_features_);
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_second_idx) * num_active_diff_features_);
            }
            if (!ORDERED) {
              PREFETCH_T0(gradients + pf_idx);
              PREFETCH_T0(hessians + pf_idx);
            }
          }
        }
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];

        pointwise_buffer.AddFirst(first_idx, gradient, hessian, stamp);
        pointwise_buffer.AddSecond(second_idx, gradient, hessian, stamp);

        if (active_diff_feature_slots_.empty()) {
          continue;
        }
        const float* first_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(first_idx) * num_active_diff_features_;
        const float* second_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(second_idx) * num_active_diff_features_;
        for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
          const float diff_value =
              first_row_diff_feature_raw_values[active_j] -
              second_row_diff_feature_raw_values[active_j];
          const uint32_t diff_bin = active_diff_feature_bin_mappers_[active_j]->ValueToBin(diff_value);
          // imitates push the bin into a single feature group (+ min_bin_ (which is always 1 for single feature groups) - offset)
          // and then extracted in a FeatureGroupIteartor (+ min_bin_ (which is 1 for FeatureGroupIterator) - offset (which is 1 for FeatureGroupIterator))
          // thus effectively should do only (+ 1 - offset)
          const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
          const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
    for (const data_size_t row_idx : first_touched) {
      const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
      const hist_t gradient = first_contributions[row_idx].gradient;
      const hist_t hessian = first_contributions[row_idx].hessian;
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + this->offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    for (const data_size_t row_idx : second_touched) {
      const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
      const hist_t gradient = second_contributions[row_idx].gradient;
      const hist_t hessian = second_contributions[row_idx].hessian;
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + this->offsets_[j] + base_offset) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }

 private:
  static constexpr int kLookupFeatureTileSize = 4;

  struct LookupPair {
    data_size_t first;
    data_size_t second;
    score_t gradient;
    score_t hessian;
  };

  struct PointwiseContribution {
    hist_t gradient;
    hist_t hessian;
  };

  template <bool CONTIGUOUS>
  void AccumulatePointwiseHistogram(const std::vector<data_size_t>& touched,
      const std::vector<PointwiseContribution>& contributions,
      const int* features, int num_features, hist_t* histogram) const {
    const bool prefetch = touched.size() >= 64 && this->offsets_.back() > 2048;
    for (data_size_t row : touched) {
      const hist_t gradient = contributions[row].gradient;
      const hist_t hessian = contributions[row].hessian;
      if (gradient == 0.0 && hessian == 0.0) { continue; }
      const auto* bins = this->data_.data() + this->RowPtr(row);
      if (prefetch) {
        // Prefetch independent feature destinations while retaining each
        // feature's original first-touch document and addition order.
        constexpr int batch_size = 16;
        size_t destinations[batch_size];
        for (int batch = 0; batch < num_features; batch += batch_size) {
          const int count = std::min(batch_size, num_features - batch);
          for (int j = 0; j < count; ++j) {
            const int f = CONTIGUOUS ? features[0] + batch + j : features[batch + j];
            destinations[j] = 2 * (static_cast<uint32_t>(bins[f]) + this->offsets_[f]);
            PREFETCH_T0(histogram + destinations[j]);
          }
          for (int j = 0; j < count; ++j) {
            histogram[destinations[j]] += gradient;
            histogram[destinations[j] + 1] += hessian;
          }
        }
      } else {
        for (int j = 0; j < num_features; ++j) {
          const int f = CONTIGUOUS ? features[0] + j : features[j];
          const size_t bin = 2 * (static_cast<uint32_t>(bins[f]) + this->offsets_[f]);
          histogram[bin] += gradient;
          histogram[bin + 1] += hessian;
        }
      }
    }
  }

  void AccumulatePointwiseHistogram(const std::vector<data_size_t>& touched,
      const std::vector<PointwiseContribution>& contributions,
      const std::vector<int>& features, int first, int end, hist_t* histogram) const {
    // Shared chunk counts can leave one side with an empty or past-end range.
    if (first >= end) { return; }
    const int count = end - first;
    // The feature list is sorted and unique, so endpoint distance identifies
    // contiguous ranges without scanning or assuming all features are used.
    if (features[end - 1] - features[first] + 1 == count) {
      AccumulatePointwiseHistogram<true>(touched, contributions,
          features.data() + first, count, histogram);
    } else {
      AccumulatePointwiseHistogram<false>(touched, contributions,
          features.data() + first, count, histogram);
    }
  }

  template <int NUM_FEATURES, bool TRACK_BINS>
  void AccumulateLookupHistograms(data_size_t start, data_size_t end,
      hist_t* const* histograms, const BIN_TYPE* tile_bins, const uint32_t* num_bins,
      const int* feature_slots, HistogramBinMask* const* masks) const {
    uint64_t tile_bin_count = 0;
    for (int j = 0; j < NUM_FEATURES; ++j) {
      tile_bin_count += static_cast<uint64_t>(num_bins[j]) * num_bins[j];
    }
    if (end - start >= 64 && tile_bin_count > 2048) {
      // Compute and prefetch a small batch of destinations before updating
      // them. The additions still follow the original row and feature order.
      constexpr int batch_size = 16;
      hist_t* destinations[batch_size][NUM_FEATURES];
      uint32_t destination_bins[batch_size][NUM_FEATURES];
      for (data_size_t batch = start; batch < end;) {
        const int count = std::min<data_size_t>(batch_size, end - batch);
        for (int i = 0; i < count; ++i) {
          const auto& pair = lookup_pairs_[batch + i];
          const auto* first = tile_bins + static_cast<size_t>(pair.first) * kLookupFeatureTileSize;
          const auto* second = tile_bins + static_cast<size_t>(pair.second) * kLookupFeatureTileSize;
          BIN_TYPE first_bins[kLookupFeatureTileSize];
          BIN_TYPE second_bins[kLookupFeatureTileSize];
          if (NUM_FEATURES == kLookupFeatureTileSize) {
            std::memcpy(first_bins, first, sizeof(first_bins));
            std::memcpy(second_bins, second, sizeof(second_bins));
          }
          for (int j = 0; j < NUM_FEATURES; ++j) {
            const uint32_t first_bin = NUM_FEATURES == kLookupFeatureTileSize ? first_bins[j] : first[feature_slots[j]];
            const uint32_t second_bin = NUM_FEATURES == kLookupFeatureTileSize ? second_bins[j] : second[feature_slots[j]];
            const uint32_t bin = first_bin * num_bins[j] + second_bin;
            auto* destination = histograms[j] + 2 * static_cast<size_t>(bin);
            destinations[i][j] = destination;
            if (TRACK_BINS) { destination_bins[i][j] = bin; }
            PREFETCH_T0(destination);
          }
        }
        for (int i = 0; i < count; ++i) {
          const auto& pair = lookup_pairs_[batch + i];
          for (int j = 0; j < NUM_FEATURES; ++j) {
            // Reuse the prefetched address and keep adjacent fields together
            // so generic compiler targets can emit packed updates.
            auto* destination = destinations[i][j];
            destination[0] += pair.gradient;
            destination[1] += pair.hessian;
            if (TRACK_BINS) { masks[j]->Mark(destination_bins[i][j]); }
          }
        }
        batch += count;
      }
      return;
    }
    for (data_size_t row = start; row < end; ++row) {
      const auto& pair = lookup_pairs_[row];
      const auto* first = tile_bins + static_cast<size_t>(pair.first) * kLookupFeatureTileSize;
      const auto* second = tile_bins + static_cast<size_t>(pair.second) * kLookupFeatureTileSize;
      BIN_TYPE first_bins[kLookupFeatureTileSize];
      BIN_TYPE second_bins[kLookupFeatureTileSize];
      if (NUM_FEATURES == kLookupFeatureTileSize) {
        std::memcpy(first_bins, first, sizeof(first_bins));
        std::memcpy(second_bins, second, sizeof(second_bins));
      }
      for (int j = 0; j < NUM_FEATURES; ++j) {
        const uint32_t first_bin = NUM_FEATURES == kLookupFeatureTileSize ? first_bins[j] : first[feature_slots[j]];
        const uint32_t second_bin = NUM_FEATURES == kLookupFeatureTileSize ? second_bins[j] : second[feature_slots[j]];
        const uint32_t bin = first_bin * num_bins[j] + second_bin;
        auto* destination = histograms[j] + 2 * static_cast<size_t>(bin);
        destination[0] += pair.gradient;
        destination[1] += pair.hessian;
        if (TRACK_BINS) { masks[j]->Mark(bin); }
      }
    }
  }

  template <bool TRACK_BINS>
  void ConstructLookupDifferentialHistogram(int tile, int tile_size, data_size_t start, data_size_t end,
      hist_t* histogram, const int8_t* is_feature_used, HistogramBinMask* const* tile_masks = nullptr) const {
    const int first_feature = tile * kLookupFeatureTileSize;
    hist_t* histograms[kLookupFeatureTileSize];
    const auto* tile_bins = lookup_feature_bins_.data() + static_cast<size_t>(first_feature) * this->num_data_;
    int feature_slots[kLookupFeatureTileSize];
    uint32_t num_bins[kLookupFeatureTileSize];
    int num_features = 0;
    HistogramBinMask* masks[kLookupFeatureTileSize];
    for (int j = 0; j < tile_size; ++j) {
      if (is_feature_used != nullptr && !is_feature_used[2 * this->num_feature_ + first_feature + j]) {
        continue;
      }
      histograms[num_features] = histogram + 2 * (active_diff_feature_hist_offsets_[first_feature + j] -
          active_diff_feature_hist_offsets_[first_feature] + active_diff_feature_bin_value_offsets_[first_feature + j]);
      feature_slots[num_features] = j;
      if (TRACK_BINS) { masks[num_features] = tile_masks[j]; }
      num_bins[num_features++] = original_feature_num_bins_[first_feature + j];
    }
    switch (num_features) {
      case 4: AccumulateLookupHistograms<4, TRACK_BINS>(start, end, histograms, tile_bins, num_bins, feature_slots, masks); break;
      case 3: AccumulateLookupHistograms<3, TRACK_BINS>(start, end, histograms, tile_bins, num_bins, feature_slots, masks); break;
      case 2: AccumulateLookupHistograms<2, TRACK_BINS>(start, end, histograms, tile_bins, num_bins, feature_slots, masks); break;
      case 1: AccumulateLookupHistograms<1, TRACK_BINS>(start, end, histograms, tile_bins, num_bins, feature_slots, masks); break;
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructLookupHistograms(const data_size_t* data_indices, data_size_t num_data,
      int num_blocks, data_size_t block_size, const score_t* gradients, const score_t* hessians,
      int num_threads, hist_t* out, const int8_t* is_feature_used, HistogramBinMask* const* bin_masks) const {
    global_timer.Start("Lookup::PointwiseHistograms");
    if (lookup_pairs_.size() < static_cast<size_t>(num_data)) {
      lookup_pairs_.resize(num_data);
    }
    lookup_block_ends_.resize(num_blocks);
    std::vector<int> pointwise_features[2];
    for (int side = 0; side < 2; ++side) {
      for (int f = 0; f < this->num_feature_; ++f) {
        if (is_feature_used == nullptr || is_feature_used[side * this->num_feature_ + f]) {
          pointwise_features[side].push_back(f);
        }
      }
    }
    // Stage pointwise contributions with the original per-block document grouping.
    const bool direct_pointwise = num_blocks < num_threads || num_blocks <= 1;
    OMP_INIT_EX();
    // Keep the same team size as the feature phases. Shrinking and regrowing
    // the OpenMP team can destroy and recreate workers between leaves.
    #pragma omp parallel for schedule(static) num_threads(num_threads) if(num_blocks > 1)
    for (int block = 0; block < num_blocks; ++block) {
      OMP_LOOP_EX_BEGIN();
      const data_size_t start = block * block_size;
      const data_size_t end = std::min(start + block_size, num_data);
      auto& buffer = pointwise_histogram_buffers_[block];
      buffer.Reset(this->num_data_, end - start);
      data_size_t nonzero_end = start;
      global_timer.Start("Lookup::StagePairsCPU");
      for (data_size_t row = start; row < end; ++row) {
        const auto idx = USE_INDICES ? data_indices[row] : row;
        const auto pair = this->paired_ranking_item_global_index_map_[idx];
        const score_t gradient = ORDERED ? gradients[row] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[row] : hessians[idx];
        // Zero contributions need no differential-bin lookup. Do not move a
        // contribution across a block boundary or change the order of the
        // remaining additions (including the pointwise document grouping).
        if (gradient != 0.0f || hessian != 0.0f) {
          lookup_pairs_[nonzero_end++] = {pair.first, pair.second, gradient, hessian};
        }
        if (!pointwise_features[0].empty()) {
          buffer.AddFirst(pair.first, gradient, hessian, buffer.stamp);
        }
        if (!pointwise_features[1].empty()) {
          buffer.AddSecond(pair.second, gradient, hessian, buffer.stamp);
        }
      }
      lookup_block_ends_[block] = nonzero_end;
      global_timer.Stop("Lookup::StagePairsCPU");
      if (!direct_pointwise) {
        // A full row-block team traverses document features once per block.
        // This has better large-leaf cache locality than feature chunks.
        global_timer.Start("Lookup::BuildPointwiseCPU");
        const uint32_t base_offset = this->offsets_.back();
        buffer.histogram.assign(4 * static_cast<size_t>(base_offset), 0.0);
        for (int side = 0; side < 2; ++side) {
          const auto& touched = side == 0 ? buffer.first_touched : buffer.second_touched;
          const auto& contributions = side == 0 ? buffer.first_contributions : buffer.second_contributions;
          auto* histogram = buffer.histogram.data() + 2 * side * base_offset;
          AccumulatePointwiseHistogram(touched, contributions,
              pointwise_features[side], 0, static_cast<int>(pointwise_features[side].size()), histogram);
        }
        global_timer.Stop("Lookup::BuildPointwiseCPU");
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    global_timer.Stop("Lookup::PointwiseHistograms");
#ifdef TIMETAG
    if (num_data == num_total_pairs_) {
      data_size_t nonzero_count = 0;
      for (int block = 0; block < num_blocks; ++block) {
        nonzero_count += lookup_block_ends_[block] - block * block_size;
      }
      Log::Info("Lookup root histogram: %d of %d pairs have nonzero contributions", nonzero_count, num_data);
    }
#endif
    const int num_tiles = (num_active_diff_features_ + kLookupFeatureTileSize - 1) / kLookupFeatureTileSize;
    // Pointwise and differential jobs own disjoint output ranges. Accumulate
    // underfilled teams' pointwise blocks directly without intermediate
    // histograms or an extra phase. Full teams merge their row-block outputs.
    constexpr int direct_feature_chunk = 32;
    const int direct_chunks = static_cast<int>((std::max(pointwise_features[0].size(),
        pointwise_features[1].size()) + direct_feature_chunk - 1) / direct_feature_chunk);
    const int pointwise_jobs = direct_chunks == 0 ? 0 : direct_pointwise ? 2 * direct_chunks : num_threads;
    const size_t pointwise_size = 2 * static_cast<size_t>(all_hist_offsets_[2 * this->num_feature_]);
    const size_t pointwise_chunk = (pointwise_size + num_threads - 1) / num_threads;
    const int num_jobs = pointwise_jobs + num_tiles;
    int64_t nonzero_pairs = 0;
    for (int block = 0; block < num_blocks; ++block) {
      nonzero_pairs += lookup_block_ends_[block] - block * block_size;
    }
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int job = 0; job < num_jobs; ++job) {
      OMP_LOOP_EX_BEGIN();
      if (job < pointwise_jobs) {
        global_timer.Start("Lookup::PointwiseHistogramsCPU");
        if (!direct_pointwise) {
          const size_t begin = std::min(pointwise_size, job * pointwise_chunk);
          const size_t end = std::min(pointwise_size, begin + pointwise_chunk);
          std::copy(pointwise_histogram_buffers_[0].histogram.data() + begin,
              pointwise_histogram_buffers_[0].histogram.data() + end, out + begin);
          for (int block = 1; block < num_blocks; ++block) {
            const auto* histogram = pointwise_histogram_buffers_[block].histogram.data();
            for (size_t b = begin; b < end; ++b) { out[b] += histogram[b]; }
          }
          global_timer.Stop("Lookup::PointwiseHistogramsCPU");
          continue;
        }
        const int side = job / direct_chunks;
        const int first = (job % direct_chunks) * direct_feature_chunk;
        const int end = std::min(first + direct_feature_chunk, static_cast<int>(pointwise_features[side].size()));
        auto* histogram = out + 2 * side * this->offsets_.back();
        for (int j = first; j < end; ++j) {
          const int feature = pointwise_features[side][j];
          std::fill(histogram + 2 * this->offsets_[feature],
              histogram + 2 * this->offsets_[feature + 1], 0.0);
        }
        for (int block = 0; block < num_blocks; ++block) {
          const auto& buffer = pointwise_histogram_buffers_[block];
          const auto& touched = side == 0 ? buffer.first_touched : buffer.second_touched;
          const auto& contributions = side == 0 ? buffer.first_contributions : buffer.second_contributions;
          AccumulatePointwiseHistogram(touched, contributions,
              pointwise_features[side], first, end, histogram);
        }
        global_timer.Stop("Lookup::PointwiseHistogramsCPU");
        continue;
      }
      const int tile = job - pointwise_jobs;
      const int f = 2 * this->num_feature_ + tile * kLookupFeatureTileSize;
      const int tile_size = std::min(kLookupFeatureTileSize, num_active_diff_features_ - tile * kLookupFeatureTileSize);
      if (is_feature_used != nullptr) {
        bool used = false;
        for (int j = 0; j < tile_size; ++j) {
          used |= is_feature_used[f + j] != 0;
        }
        if (!used) {
          continue;
        }
      }
      const size_t histogram_size = 2 * (all_hist_offsets_[f + tile_size] - all_hist_offsets_[f]);
      global_timer.Start("Lookup::DifferentialHistogramsCPU");
      hist_t* output = out + 2 * all_hist_offsets_[f];
      bool track_bins = bin_masks != nullptr;
      for (int j = 0; j < tile_size && track_bins; ++j) {
        if (is_feature_used == nullptr || is_feature_used[f + j]) {
          track_bins = bin_masks[f + j] != nullptr &&
              all_hist_offsets_[f + j + 1] - all_hist_offsets_[f + j] ==
              original_feature_num_bins_[tile * kLookupFeatureTileSize + j] *
              original_feature_num_bins_[tile * kLookupFeatureTileSize + j];
        }
      }
      // Dense leaves repeatedly touch the same support bits. Rebuild support
      // once from the completed histogram instead of marking every update.
      bool rebuild_masks = track_bins && num_blocks > 0;
      uint32_t largest_bin_count = 0;
      for (int j = 0; j < tile_size && rebuild_masks; ++j) {
        rebuild_masks = is_feature_used == nullptr || is_feature_used[f + j];
        largest_bin_count = std::max(largest_bin_count, all_hist_offsets_[f + j + 1] - all_hist_offsets_[f + j]);
      }
      rebuild_masks &= nonzero_pairs >= 2 * static_cast<int64_t>(largest_bin_count);
      if (track_bins && !rebuild_masks) {
        HistogramBinMask* output_masks[kLookupFeatureTileSize] = {};
        global_timer.Start("Lookup::SparseClearCPU");
        for (int j = 0; j < tile_size; ++j) {
          if (is_feature_used != nullptr && !is_feature_used[f + j]) { continue; }
          const size_t offset = all_hist_offsets_[f + j] - all_hist_offsets_[f];
          const int bins = all_hist_offsets_[f + j + 1] - all_hist_offsets_[f + j];
          output_masks[j] = bin_masks[f + j];
          output_masks[j]->Clear(output + 2 * offset, bins);
        }
        global_timer.Stop("Lookup::SparseClearCPU");
        for (int block = 0; block < num_blocks; ++block) {
          const data_size_t start = block * block_size;
          const data_size_t end = lookup_block_ends_[block];
          if (start == end) { continue; }
          global_timer.Start("Lookup::AccumulateTrackedCPU");
          ConstructLookupDifferentialHistogram<true>(tile, tile_size, start, end,
              output, is_feature_used, output_masks);
          global_timer.Stop("Lookup::AccumulateTrackedCPU");
        }
        global_timer.Stop("Lookup::DifferentialHistogramsCPU");
        continue;
      }
      if (bin_masks != nullptr) {
        for (int j = 0; j < tile_size; ++j) {
          if (bin_masks[f + j] != nullptr) { bin_masks[f + j]->Invalidate(); }
        }
      }
      global_timer.Start("Lookup::ClearCPU");
      std::fill(output, output + histogram_size, 0.0);
      global_timer.Stop("Lookup::ClearCPU");
      // Accumulate directly in input order across row blocks. This removes
      // block scratch histograms and reductions; only the parenthesization
      // of floating-point sums changes from the old blockwise interface.
      for (int block = 0; block < num_blocks; ++block) {
        {
          const data_size_t start = block * block_size;
          const data_size_t end = lookup_block_ends_[block];
          global_timer.Start("Lookup::AccumulateDifferentialCPU");
          ConstructLookupDifferentialHistogram<false>(tile, tile_size, start, end, output, is_feature_used);
          global_timer.Stop("Lookup::AccumulateDifferentialCPU");
        }
      }
      if (rebuild_masks) {
        global_timer.Start("Lookup::RebuildMasksCPU");
        for (int j = 0; j < tile_size; ++j) {
          const size_t offset = all_hist_offsets_[f + j] - all_hist_offsets_[f];
          const int bins = all_hist_offsets_[f + j + 1] - all_hist_offsets_[f + j];
          bin_masks[f + j]->FromHistogram(output + 2 * offset, bins);
        }
        global_timer.Stop("Lookup::RebuildMasksCPU");
      }
      global_timer.Stop("Lookup::DifferentialHistogramsCPU");
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
#ifdef TIMETAG
    if (num_data == num_total_pairs_) {
      size_t occupied = 0;
      const size_t first_bin = all_hist_offsets_[2 * this->num_feature_];
      for (size_t b = first_bin; b < all_hist_offsets_.back(); ++b) {
        occupied += out[2 * b] != 0.0 || out[2 * b + 1] != 0.0;
      }
      Log::Info("Lookup root differential histogram: %zu of %zu bins occupied", occupied,
          static_cast<size_t>(all_hist_offsets_.back()) - first_bin);
    }
#endif
  }

  struct PointwiseHistogramBuffer {
    // Both fields are read and updated together. Keep them adjacent without
    // changing document first-touch order or per-document addition order.
    std::vector<PointwiseContribution> first_contributions;
    std::vector<PointwiseContribution> second_contributions;
    std::vector<uint32_t> first_stamps;
    std::vector<uint32_t> second_stamps;
    std::vector<data_size_t> first_touched;
    std::vector<data_size_t> second_touched;
    std::vector<hist_t> histogram;
    uint32_t stamp = 0;

    void Reset(data_size_t num_data, size_t reserve_size) {
      const size_t size = static_cast<size_t>(num_data);
      if (first_contributions.size() != size) {
        first_contributions.assign(size, {0.0, 0.0});
        second_contributions.assign(size, {0.0, 0.0});
        first_stamps.assign(size, 0);
        second_stamps.assign(size, 0);
        stamp = 0;
      } else if (stamp == std::numeric_limits<uint32_t>::max()) {
        std::fill(first_stamps.begin(), first_stamps.end(), 0);
        std::fill(second_stamps.begin(), second_stamps.end(), 0);
        stamp = 0;
      }
      ++stamp;
      first_touched.clear();
      second_touched.clear();
      if (first_touched.capacity() < reserve_size) {
        first_touched.reserve(reserve_size);
      }
      if (second_touched.capacity() < reserve_size) {
        second_touched.reserve(reserve_size);
      }
    }

    void AddFirst(data_size_t idx, score_t gradient, score_t hessian, uint32_t current_stamp) {
      if (first_stamps[idx] != current_stamp) {
        first_stamps[idx] = current_stamp;
        first_contributions[idx] = {0.0, 0.0};
        first_touched.push_back(idx);
      }
      first_contributions[idx].gradient += gradient;
      first_contributions[idx].hessian += hessian;
    }

    void AddSecond(data_size_t idx, score_t gradient, score_t hessian, uint32_t current_stamp) {
      if (second_stamps[idx] != current_stamp) {
        second_stamps[idx] = current_stamp;
        second_contributions[idx] = {0.0, 0.0};
        second_touched.push_back(idx);
      }
      second_contributions[idx].gradient += gradient;
      second_contributions[idx].hessian += hessian;
    }
  };

  void InitDiffFeatureRawValueBuffer(data_size_t num_data) {
    num_active_diff_features_ = static_cast<int>(active_diff_feature_slots_.size());
    if (num_active_diff_features_ == 0 || differential_feature_bin_mode_ == "lookup") {
      return;
    }
    diff_feature_raw_values_.resize(
        static_cast<size_t>(num_data) * num_active_diff_features_);
    const int num_threads = OMP_NUM_THREADS();
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data; ++i) {
      float* row_diff_feature_raw_values =
          diff_feature_raw_values_.data() +
          static_cast<size_t>(i) * num_active_diff_features_;
      for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
        const int j = active_diff_feature_slots_[active_j];
        const float* feature_values = diff_feature_raw_data_ptrs_[j];
        row_diff_feature_raw_values[active_j] = feature_values[i];
      }
    }
  }

  std::vector<std::unique_ptr<const BinMapper>> diff_feature_bin_mappers_;
  std::vector<int> diff_feature_to_original_feature_slot_;
  std::vector<int> active_diff_feature_to_original_feature_slot_;
  std::vector<uint32_t> diff_feature_hist_offsets_;
  std::vector<uint32_t> diff_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_most_freq_bins_;
  std::vector<uint32_t> original_feature_num_bins_;
  std::vector<BIN_TYPE> lookup_feature_bins_;
  std::vector<const float*> diff_feature_raw_data_ptrs_;
  std::vector<float> diff_feature_raw_values_;
  std::vector<int> active_diff_feature_slots_;
  std::vector<uint32_t> active_diff_feature_hist_offsets_;
  std::vector<uint32_t> active_diff_feature_bin_value_offsets_;
  std::vector<const BinMapper*> active_diff_feature_bin_mappers_;
  mutable std::vector<PointwiseHistogramBuffer> pointwise_histogram_buffers_;
  mutable std::vector<LookupPair> lookup_pairs_;
  mutable std::vector<data_size_t> lookup_block_ends_;
  mutable data_size_t num_total_pairs_ = 0;
  std::vector<uint32_t> all_hist_offsets_;
  int num_diff_features_;
  int num_active_diff_features_ = 0;
  bool use_fast_pairwise_bin_lookup_;
  std::string differential_feature_bin_mode_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
