/*!
 * Copyright (c) 2025-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2025-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/metric/map_metric.hpp"

class MapMetricTest : public testing::Test {};

TEST_F(MapMetricTest, CalMapAtK) {
  std::vector<double> out(2);
  LightGBM::MapMetric::CalMapAtK({2, 5}, 2,
                                 new LightGBM::label_t[]{0.4f, 0.6f, 0.8f, 0.2f}, new double[]{1., 3., 2., 4.},
                                 4, &out);
  EXPECT_NEAR(out[0], 0.25, 1e-6;
  EXPECT_NEAR(out[1], 7 / 12., 1e-6;
  LightGBM::MapMetric::CalMapAtK({2, 5}, 2,
                                 new LightGBM::label_t[]{0.1f, 0.9f, 0.8f}, new double[]{6., 1., 2.},
                                 3, &out);
  EXPECT_NEAR(out[0], 0.25, 1e-6);
  EXPECT_NEAR(out[1], 7 / 12., 1e-6);
}

TEST_F(MapMetricTest, Eval) {
  LightGBM::MapMetric metric = LightGBM::MapMetric(LightGBM::Config(
    std::unordered_map<std::string, std::string>{{"eval_at", "2,5"}}
  ));
  // 7 documents across 2 queries: query 0 has 4 docs, query 1 has 3 docs
  constexpr LightGBM::data_size_t num_data = 9;
  LightGBM::Metadata metadata;
  metadata.Init(num_data, -1, -1);
  metadata.SetQuery(new LightGBM::data_size_t[]{4, 3, 2}, 3);
  metadata.SetLabel(new LightGBM::label_t[]{
                      0.4f, 0.6f, 0.8f, 0.2f,  // query 0
                      0.1f, 0.9f, 0.8f,  // query 1
                      0.4f, 0.3f,  // query 2
                    },
                    num_data);
  metric.Init(metadata, num_data);
  std::vector<double> out = metric.Eval(new double[]{
                                          1., 3., 2., 4.,  // query 0
                                          6., 1., 2.,  // query 1
                                          3., 4.  // query 2
                                        },
                                        nullptr);
  EXPECT_NEAR(out[0], 0.25, 1e-6);
  EXPECT_NEAR(out[1], 7 / 12., 1e-6);
}
