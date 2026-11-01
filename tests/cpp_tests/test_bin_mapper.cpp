/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/bin.h>

#include <limits>
#include <vector>

using LightGBM::BinMapper;
using LightGBM::BinType;
using LightGBM::MissingType;

namespace {
constexpr int kMaxBin = 255;
constexpr int kMinDataInBin = 1;
constexpr int kMinSplitData = 20;

std::vector<double> AllMissingSample(int num_values) {
  return std::vector<double>(static_cast<size_t>(num_values),
                             std::numeric_limits<double>::quiet_NaN());
}

}  // namespace

TEST(BinMapperTest, AllMissingCategoricalIsConsistentForSerialization) {
  // a sampled categorical column where every value is missing
  const size_t total_sample_cnt = 100;
  const int num_sample_values = 100;
  std::vector<double> values = AllMissingSample(num_sample_values);

  BinMapper bin_mapper;
  bin_mapper.FindBin(values.data(), num_sample_values, total_sample_cnt,
                     kMaxBin, kMinDataInBin, kMinSplitData, /* pre_filter */ true,
                     BinType::CategoricalBin, /* use_missing */ true,
                     /* zero_as_missing */ false, std::vector<double>{});

  ASSERT_EQ(1, bin_mapper.num_bin());
  ASSERT_EQ(BinType::CategoricalBin, bin_mapper.bin_type());
  ASSERT_EQ(MissingType::NaN, bin_mapper.missing_type());
  ASSERT_TRUE(bin_mapper.is_trivial());

  std::vector<char> buffer(bin_mapper.SizesInByte());
  bin_mapper.CopyTo(buffer.data());

  BinMapper deserialized(buffer.data());
  EXPECT_EQ(1, deserialized.num_bin());
  EXPECT_EQ(MissingType::NaN, deserialized.missing_type());
  EXPECT_TRUE(deserialized.is_trivial());
  // the only bin holds the dummy category used for missing values,
  // exactly like the normal path does
  EXPECT_EQ(-1, static_cast<int>(deserialized.BinToValue(0)));
  EXPECT_EQ(0, deserialized.GetDefaultBin());
  EXPECT_EQ(0, deserialized.GetMostFreqBin());
}

TEST(BinMapperTest, AllMissingNumericalIsConsistentForSerialization) {
  // control case: same setup with a numerical feature,
  // which already produced a consistent mapper before the fix
  const size_t total_sample_cnt = 100;
  const int num_sample_values = 100;
  std::vector<double> values = AllMissingSample(num_sample_values);

  BinMapper bin_mapper;
  bin_mapper.FindBin(values.data(), num_sample_values, total_sample_cnt,
                     kMaxBin, kMinDataInBin, kMinSplitData, /* pre_filter */ true,
                     BinType::NumericalBin, /* use_missing */ true,
                     /* zero_as_missing */ false, std::vector<double>{});

  std::vector<char> buffer(bin_mapper.SizesInByte());
  bin_mapper.CopyTo(buffer.data());

  BinMapper deserialized(buffer.data());
  EXPECT_EQ(bin_mapper.num_bin(), deserialized.num_bin());
  EXPECT_EQ(bin_mapper.missing_type(), deserialized.missing_type());
  EXPECT_EQ(bin_mapper.is_trivial(), deserialized.is_trivial());
  EXPECT_EQ(0, bin_mapper.GetMostFreqBin());
  EXPECT_EQ(0, deserialized.GetMostFreqBin());
}
