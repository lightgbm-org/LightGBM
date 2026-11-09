/* Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */

#include <LightGBM/c_api.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_API(call) do { \
  if ((call) != 0) { \
    fprintf(stderr, "%s: %s\n", #call, LGBM_GetLastError()); \
    goto cleanup; \
  } \
} while (0)

int main(void) {
  const float data[] = {0.0f, 0.0f, 1.0f, 1.0f};
  const float labels[] = {0.0f, 0.0f, 1.0f, 1.0f};
  const char* parameters = "objective=regression num_leaves=2 learning_rate=1 "
                           "min_data_in_leaf=1 min_data_in_bin=1 num_threads=1 verbosity=-1";
  DatasetHandle dataset = NULL;
  BoosterHandle booster = NULL;
  double predictions[4] = {0.0, 0.0, 0.0, 0.0};
  int64_t out_len = 0;
  int num_rows = 0;
  int num_features = 0;
  int is_finished = 0;
  int status = EXIT_FAILURE;
  int i;

  CHECK_API(LGBM_DatasetCreateFromMat(data, C_API_DTYPE_FLOAT32, 4, 1, 1,
                                    parameters, NULL, &dataset));
  CHECK_API(LGBM_DatasetGetNumData(dataset, &num_rows));
  CHECK_API(LGBM_DatasetGetNumFeature(dataset, &num_features));
  if (num_rows != 4 || num_features != 1) {
    fprintf(stderr, "Unexpected dataset dimensions: %d x %d\n", num_rows, num_features);
    goto cleanup;
  }
  CHECK_API(LGBM_DatasetSetField(dataset, "label", labels, 4, C_API_DTYPE_FLOAT32));
  CHECK_API(LGBM_BoosterCreate(dataset, parameters, &booster));
  CHECK_API(LGBM_BoosterUpdateOneIter(booster, &is_finished));
  CHECK_API(LGBM_BoosterPredictForMat(booster, data, C_API_DTYPE_FLOAT32, 4, 1, 1,
                                    C_API_PREDICT_NORMAL, 0, -1, "num_threads=1", &out_len, predictions));
  if (out_len != 4) {
    fprintf(stderr, "Unexpected prediction count: %" PRId64 "\n", out_len);
    goto cleanup;
  }
  for (i = 0; i < 4; ++i) {
    if (!isfinite(predictions[i]) || fabs(predictions[i] - labels[i]) > 1e-6) {
      fprintf(stderr, "Prediction %d: expected %g, got %g\n", i, labels[i], predictions[i]);
      goto cleanup;
    }
  }
  status = EXIT_SUCCESS;

cleanup:
  if (booster != NULL && LGBM_BoosterFree(booster) != 0) {
    fprintf(stderr, "LGBM_BoosterFree: %s\n", LGBM_GetLastError());
    status = EXIT_FAILURE;
  }
  if (dataset != NULL && LGBM_DatasetFree(dataset) != 0) {
    fprintf(stderr, "LGBM_DatasetFree: %s\n", LGBM_GetLastError());
    status = EXIT_FAILURE;
  }
  return status;
}
