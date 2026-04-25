/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/c_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

using LightGBM::TestUtils;

void test_predict_type(int predict_type, int num_predicts) {
    // Load some test data
    int result;

    DatasetHandle train_dataset;
    result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.train", "max_bin=15", &train_dataset);
    EXPECT_EQ(0, result) << "LoadDatasetFromExamples train result code: " << result;

    BoosterHandle booster_handle;
    result = LGBM_BoosterCreate(train_dataset, "app=binary metric=auc num_leaves=31 verbose=0", &booster_handle);
    EXPECT_EQ(0, result) << "LGBM_BoosterCreate result code: " << result;

    for (int i = 0; i < 51; i++) {
        int produced_empty_tree;
        result = LGBM_BoosterUpdateOneIter(
            booster_handle,
            &produced_empty_tree);
        EXPECT_EQ(0, result) << "LGBM_BoosterUpdateOneIter result code: " << result;
    }

    int n_features;
    result = LGBM_BoosterGetNumFeature(
        booster_handle,
        &n_features);
    EXPECT_EQ(0, result) << "LGBM_BoosterGetNumFeature result code: " << result;
    EXPECT_EQ(28, n_features) << "LGBM_BoosterGetNumFeature number of features: " << n_features;

    // Run a single row prediction and compare with regular Mat prediction:
    int64_t output_size;
    result = LGBM_BoosterCalcNumPredict(
        booster_handle,
        1,
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        &output_size);
    EXPECT_EQ(0, result) << "LGBM_BoosterCalcNumPredict result code: " << result;
    EXPECT_EQ(num_predicts, output_size) << "LGBM_BoosterCalcNumPredict output size: " << output_size;

    std::ifstream test_file("examples/binary_classification/binary.test");
    std::vector<double> test;
    double x;
    int test_set_size = 0;
    while (test_file >> x) {
        if (test_set_size % (n_features + 1) == 0) {
            // Drop the result from the dataset, we only care about checking that prediction results are equal
            // in both cases
            test_file >> x;
            test_set_size++;
        }
        test.push_back(x);
        test_set_size++;
    }
    EXPECT_EQ(test_set_size % (n_features + 1), 0) << "Test size mismatch with dataset size (%)";
    test_set_size /= (n_features + 1);
    EXPECT_EQ(test_set_size, 500) << "Improperly parsed test file (test_set_size)";
    EXPECT_EQ(test.size(), test_set_size * n_features) << "Improperly parsed test file (test len)";

    std::vector<double> mat_output(output_size * test_set_size, -1);
    int64_t written;
    result = LGBM_BoosterPredictForMat(
        booster_handle,
        &test[0],
        C_API_DTYPE_FLOAT64,
        test_set_size,         // nrow
        n_features,            // ncol
        1,                     // is_row_major
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        "",
        &written,
        &mat_output[0]);
    EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMat result code: " << result;

    // Test LGBM_BoosterPredictForMat in multi-threaded mode
    const int kNThreads = 10;
    const int numIterations = 5;
    std::vector<std::thread> predict_for_mat_threads(kNThreads);
    for (int i = 0; i < kNThreads; i++) {
        predict_for_mat_threads[i] = std::thread(
            [
                i, test_set_size, output_size, n_features,
                    test = &test[0], booster_handle, predict_type, numIterations
            ]() {
                for (int j = 0; j < numIterations; j++) {
                    int result;
                    std::vector<double> mat_output(output_size * test_set_size, -1);
                    int64_t written;
                    result = LGBM_BoosterPredictForMat(
                        booster_handle,
                        &test[0],
                        C_API_DTYPE_FLOAT64,
                        test_set_size,         // nrow
                        n_features,            // ncol
                        1,                     // is_row_major
                        predict_type,          // predict_type
                        0,                     // start_iteration
                        -1,                    // num_iteration
                        "",
                        &written,
                        &mat_output[0]);
                    EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMat result code: " << result;
                }
            });
    }
    for (std::thread& t : predict_for_mat_threads) {
        t.join();
    }

    // Now let's run with the single row fast prediction API:
    FastConfigHandle fast_configs[kNThreads];
    for (int i = 0; i < kNThreads; i++) {
        result = LGBM_BoosterPredictForMatSingleRowFastInit(
            booster_handle,
            predict_type,          // predict_type
            0,                     // start_iteration
            -1,                    // num_iteration
            C_API_DTYPE_FLOAT64,
            n_features,
            "",
            &fast_configs[i]);
        EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMatSingleRowFastInit result code: " << result;
    }

    std::vector<double> single_row_output(output_size * test_set_size, -1);
    std::vector<std::thread> single_row_threads(kNThreads);
    int batch_size = (test_set_size + kNThreads - 1) / kNThreads;  // round up
    for (int i = 0; i < kNThreads; i++) {
        single_row_threads[i] = std::thread(
            [
                i, batch_size, test_set_size, output_size, n_features,
                    test = &test[0], fast_configs = &fast_configs[0], single_row_output = &single_row_output[0]
            ]() {
                int result;
                int64_t written;
                for (int j = i * batch_size; j < std::min((i + 1) * batch_size, test_set_size); j++) {
                    result = LGBM_BoosterPredictForMatSingleRowFast(
                        fast_configs[i],
                        &test[j * n_features],
                        &written,
                        &single_row_output[j * output_size]);
                    EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMatSingleRowFast result code: " << result;
                    EXPECT_EQ(written, output_size) << "LGBM_BoosterPredictForMatSingleRowFast unexpected written output size";
                }
            });
      }
    for (std::thread& t : single_row_threads) {
        t.join();
    }

    EXPECT_EQ(single_row_output, mat_output) << "LGBM_BoosterPredictForMatSingleRowFast output mismatch with LGBM_BoosterPredictForMat";

    // Free all:
    for (int i = 0; i < kNThreads; i++) {
        result = LGBM_FastConfigFree(fast_configs[i]);
        EXPECT_EQ(0, result) << "LGBM_FastConfigFree result code: " << result;
    }

    result = LGBM_BoosterFree(booster_handle);
    EXPECT_EQ(0, result) << "LGBM_BoosterFree result code: " << result;

    result = LGBM_DatasetFree(train_dataset);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}

TEST(SingleRow, Normal) {
    test_predict_type(C_API_PREDICT_NORMAL, 1);
}

TEST(SingleRow, Contrib) {
    test_predict_type(C_API_PREDICT_CONTRIB, 29);
}
