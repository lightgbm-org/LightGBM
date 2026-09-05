/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../src/treelearner/split_info.hpp"

namespace {

LightGBM::SplitInfo MakeSplit(int num_cat_threshold) {
  LightGBM::SplitInfo split;
  split.feature = 7;
  split.threshold = 11;
  split.left_count = 101;
  split.right_count = 303;
  split.left_output = -0.75;
  split.right_output = 1.25;
  split.gain = 4.5;
  split.left_sum_gradient = -13.0;
  split.left_sum_hessian = 0.75;
  split.right_sum_gradient = 23.0;
  split.right_sum_hessian = 1.5;
  split.left_sum_gradient_and_hessian = -(INT64_C(1) << 48) - 17;
  split.right_sum_gradient_and_hessian = (INT64_C(1) << 49) + 31;
  split.default_left = false;
  split.monotone_type = -1;
  split.num_cat_threshold = num_cat_threshold;
  for (int i = 0; i < num_cat_threshold; ++i) {
    split.cat_threshold.push_back(static_cast<uint32_t>(i * 7 + 3));
  }
  return split;
}

void ExpectSameSplit(const LightGBM::SplitInfo& expected, const LightGBM::SplitInfo& actual) {
  EXPECT_EQ(expected.feature, actual.feature);
  EXPECT_EQ(expected.threshold, actual.threshold);
  EXPECT_EQ(expected.left_count, actual.left_count);
  EXPECT_EQ(expected.right_count, actual.right_count);
  EXPECT_EQ(expected.left_output, actual.left_output);
  EXPECT_EQ(expected.right_output, actual.right_output);
  EXPECT_EQ(expected.gain, actual.gain);
  EXPECT_EQ(expected.left_sum_gradient, actual.left_sum_gradient);
  EXPECT_EQ(expected.left_sum_hessian, actual.left_sum_hessian);
  EXPECT_EQ(expected.right_sum_gradient, actual.right_sum_gradient);
  EXPECT_EQ(expected.right_sum_hessian, actual.right_sum_hessian);
  EXPECT_EQ(expected.left_sum_gradient_and_hessian, actual.left_sum_gradient_and_hessian);
  EXPECT_EQ(expected.right_sum_gradient_and_hessian, actual.right_sum_gradient_and_hessian);
  EXPECT_EQ(expected.default_left, actual.default_left);
  EXPECT_EQ(expected.monotone_type, actual.monotone_type);
  EXPECT_EQ(expected.num_cat_threshold, actual.num_cat_threshold);
  EXPECT_EQ(expected.cat_threshold, actual.cat_threshold);
}

void CheckSerialization(int max_cat_threshold, int num_cat_threshold) {
  SCOPED_TRACE(::testing::Message() << "max_cat_threshold=" << max_cat_threshold
                                  << ", num_cat_threshold=" << num_cat_threshold);
  const auto split = MakeSplit(num_cat_threshold);
  const auto size = LightGBM::SplitInfo::Size(max_cat_threshold);
  const int guard_size = 32;
  const char sentinel = '\x5a';
  std::vector<char> storage(size + 2 * guard_size, sentinel);
  split.CopyTo(storage.data() + guard_size);

  EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + guard_size,
                          [sentinel](char value) { return value == sentinel; }));
  EXPECT_TRUE(std::all_of(storage.begin() + guard_size + size, storage.end(),
                          [sentinel](char value) { return value == sentinel; }));

  LightGBM::SplitInfo restored;
  restored.CopyFrom(storage.data() + guard_size);
  ExpectSameSplit(split, restored);

  restored.cat_threshold.assign(max_cat_threshold + 1, UINT32_MAX);
  restored.CopyFrom(storage.data() + guard_size);
  ExpectSameSplit(split, restored);
}

}  // namespace

TEST(SplitInfoSerialization, NumericalRoundTrip) {
  CheckSerialization(0, 0);
  CheckSerialization(32, 0);
}

TEST(SplitInfoSerialization, CategoricalRoundTrip) {
  for (int max_cat_threshold : {1, 28, 29, 32, 64}) {
    CheckSerialization(max_cat_threshold, 1);
    CheckSerialization(max_cat_threshold, max_cat_threshold);
  }
  CheckSerialization(32, 28);
  CheckSerialization(32, 29);
}
