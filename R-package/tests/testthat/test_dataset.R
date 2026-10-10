data(agaricus.train, package = "lightgbm")
train_data <- agaricus.train$data[seq_len(1000L), ]
train_label <- agaricus.train$label[seq_len(1000L)]

data(agaricus.test, package = "lightgbm")
test_data <- agaricus.test$data[1L:100L, ]
test_label <- agaricus.test$label[1L:100L]

test_that("lgb.Dataset: basic construction, saving, loading", {
  # from sparse matrix
  dtest1 <- lgb.Dataset(
    test_data
    , label = test_label
    , params = list(
      verbose = .LGB_VERBOSITY
    )
  )
  # from dense matrix
  dtest2 <- lgb.Dataset(as.matrix(test_data), label = test_label)
  expect_equal(get_field(dtest1, "label"), get_field(dtest2, "label"))

  # save to a local file
  tmp_file <- tempfile("lgb.Dataset_")
  lgb.Dataset.save(dtest1, tmp_file)
  # read from a local file
  dtest3 <- lgb.Dataset(
    tmp_file
    , params = list(
      verbose = .LGB_VERBOSITY
    )
  )
  lgb.Dataset.construct(dtest3)
  unlink(tmp_file)
  expect_equal(get_field(dtest1, "label"), get_field(dtest3, "label"))
})

test_that("lgb.Dataset: get_field & set_field", {
  dtest <- lgb.Dataset(test_data)
  dtest$construct()

  set_field(dtest, "label", test_label)
  labels <- get_field(dtest, "label")
  expect_equal(test_label, get_field(dtest, "label"))

  expect_true(length(get_field(dtest, "weight")) == 0L)
  expect_true(length(get_field(dtest, "init_score")) == 0L)

  # any other label should error
  expect_error(
    set_field(dtest, "asdf", test_label)
    , regexp = "Dataset$set_field(): field_name must be one of the following: 'label', 'weight', 'init_score', 'group'"  # nolint: line_length.
    , fixed = TRUE
  )
})

test_that("lgb.Dataset: slice, dim", {
  dtest <- lgb.Dataset(test_data, label = test_label)
  lgb.Dataset.construct(dtest)
  expect_equal(dim(dtest), dim(test_data))
  dsub1 <- lgb.slice.Dataset(dtest, seq_len(42L))
  lgb.Dataset.construct(dsub1)
  expect_equal(nrow(dsub1), 42L)
  expect_equal(ncol(dsub1), ncol(test_data))
})

test_that("Dataset$set_reference() on a constructed Dataset fails if raw data has been freed", {
  dtrain <- lgb.Dataset(train_data, label = train_label)
  dtrain$construct()
  dtest <- lgb.Dataset(test_data, label = test_label)
  dtest$construct()
  expect_error({
    dtest$set_reference(dtrain)
  }, regexp = "cannot set reference after freeing raw data")
})

test_that("Dataset$set_reference() fails if reference is not a Dataset", {
  dtrain <- lgb.Dataset(
    train_data
    , label = train_label
    , free_raw_data = FALSE
  )
  expect_error({
    dtrain$set_reference(reference = data.frame(x = rnorm(10L)))
  }, regexp = "Can only use lgb.Dataset as a reference")

  # passing NULL when the Dataset already has a reference raises an error
  dtest <- lgb.Dataset(
    test_data
    , label = test_label
    , free_raw_data = FALSE
  )
  dtrain$set_reference(dtest)
  expect_error({
    dtrain$set_reference(reference = NULL)
  }, regexp = "Can only use lgb.Dataset as a reference")
})

test_that("Dataset$set_reference() setting reference to the same Dataset has no side effects", {
  dtrain <- lgb.Dataset(
    train_data
    , label = train_label
    , free_raw_data = FALSE
    , categorical_feature = c(2L, 3L)
  )
  dtrain$construct()

  cat_features_before <- dtrain$.__enclos_env__$private$categorical_feature
  colnames_before <- dtrain$get_colnames()
  predictor_before <- dtrain$.__enclos_env__$private$predictor

  dtrain$set_reference(dtrain)
  expect_identical(
    cat_features_before
    , dtrain$.__enclos_env__$private$categorical_feature
  )
  expect_identical(
    colnames_before
    , dtrain$get_colnames()
  )
  expect_identical(
    predictor_before
    , dtrain$.__enclos_env__$private$predictor
  )
})

test_that("Dataset$set_reference() updates categorical_feature, colnames, and predictor", {
  dtrain <- lgb.Dataset(
    train_data
    , label = train_label
    , free_raw_data = FALSE
    , categorical_feature = c(2L, 3L)
  )
  dtrain$construct()
  bst <- Booster$new(
    train_set = dtrain
    , params = list(verbose = -1L, num_threads = .LGB_MAX_THREADS)
  )
  dtrain$.__enclos_env__$private$predictor <- bst$to_predictor()

  test_original_feature_names <- paste0("feature_col_", seq_len(ncol(test_data)))
  dtest <- lgb.Dataset(
    test_data
    , label = test_label
    , free_raw_data = FALSE
    , colnames = test_original_feature_names
  )
  dtest$construct()

  # at this point, dtest should not have categorical_feature
  expect_null(dtest$.__enclos_env__$private$predictor)
  expect_null(dtest$.__enclos_env__$private$categorical_feature)
  expect_identical(
    dtest$get_colnames()
    , test_original_feature_names
  )

  dtest$set_reference(dtrain)

  # after setting reference to dtrain, those attributes should have dtrain's values
  expect_true(methods::is(
    dtest$.__enclos_env__$private$predictor
    , "lgb.Predictor"
  ))
  expect_identical(
    dtest$.__enclos_env__$private$predictor$.__enclos_env__$private$handle
    , dtrain$.__enclos_env__$private$predictor$.__enclos_env__$private$handle
  )
  expect_identical(
    dtest$.__enclos_env__$private$categorical_feature
    , dtrain$.__enclos_env__$private$categorical_feature
  )
  expect_identical(
    dtest$get_colnames()
    , dtrain$get_colnames()
  )
  expect_false(
    identical(dtest$get_colnames(), test_original_feature_names)
  )
})

test_that("lgb.Dataset: colnames", {
  dtest <- lgb.Dataset(test_data, label = test_label)
  expect_equal(colnames(dtest), colnames(test_data))
  lgb.Dataset.construct(dtest)
  expect_equal(colnames(dtest), colnames(test_data))
  expect_error({
    colnames(dtest) <- "asdf"
  }, regexp = "can't assign '1' colnames to an lgb.Dataset with '126' columns")
  new_names <- make.names(seq_len(ncol(test_data)))
  expect_silent({
    colnames(dtest) <- new_names
  })
  expect_equal(colnames(dtest), new_names)
})

test_that("lgb.Dataset: nrow is correct for a very sparse matrix", {
  nr <- 1000L
  x <- Matrix::rsparsematrix(nr, 100L, density = 0.0005)
  # we want it very sparse, so that last rows are empty
  expect_lt(max(x@i), nr)
  dtest <- lgb.Dataset(x)
  expect_equal(dim(dtest), dim(x))
})

test_that("lgb.Dataset: Dataset should be able to construct from matrix and return non-null handle", {
  rawData <- matrix(runif(1000L), ncol = 10L)
  ref_handle <- NULL
  handle <- .Call(
    LGBM_DatasetCreateFromMat_R
    , rawData
    , nrow(rawData)
    , ncol(rawData)
    , lightgbm:::.params2str(params = list())
    , ref_handle
  )
  expect_true(methods::is(handle, "externalptr"))
  expect_false(is.null(handle))
  .Call(LGBM_DatasetFree_R, handle)
  handle <- NULL
})

test_that("cpp errors should be raised as proper R errors", {
  testthat::skip_if(
    Sys.getenv("COMPILER", "") == "MSVC"
    , message = "Skipping on Visual Studio"
  )
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , init_score = seq_len(10L)
  )
  expect_error({
    capture.output({
      dtrain$construct()
    }, type = "message")
  }, regexp = "Initial score size doesn't match data size")
})

test_that("lgb.Dataset$set_field() should convert 'group' to integer", {
  ds <- lgb.Dataset(
    data = matrix(rnorm(100L), nrow = 50L, ncol = 2L)
    , label = sample(c(0L, 1L), size = 50L, replace = TRUE)
  )
  ds$construct()
  current_group <- ds$get_field("group")
  expect_null(current_group)
  group_as_numeric <- rep(25.0, 2L)
  ds$set_field("group", group_as_numeric)
  expect_identical(ds$get_field("group"), as.integer(group_as_numeric))
})

test_that("lgb.Dataset should throw an error if 'reference' is provided but of the wrong format", {
  data(agaricus.test, package = "lightgbm")
  test_data <- agaricus.test$data[1L:100L, ]
  test_label <- agaricus.test$label[1L:100L]
  # Try to trick lgb.Dataset() into accepting bad input
  expect_error({
    dtest <- lgb.Dataset(
      data = test_data
      , label = test_label
      , reference = data.frame(x = seq_len(10L), y = seq_len(10L))
    )
  }, regexp = "reference must be a")
})

test_that("Dataset$new() should throw an error if 'predictor' is provided but of the wrong format", {
  data(agaricus.test, package = "lightgbm")
  test_data <- agaricus.test$data[1L:100L, ]
  test_label <- agaricus.test$label[1L:100L]
  expect_error({
    dtest <- Dataset$new(
      data = test_data
      , label = test_label
      , predictor = data.frame(x = seq_len(10L), y = seq_len(10L))
    )
  }, regexp = "predictor must be a", fixed = TRUE)
})

test_that("Dataset$get_params() successfully returns parameters if you passed them", {
  # note that this list uses one "main" parameter (feature_pre_filter) and one that
  # is an alias (is_sparse), to check that aliases are handled correctly
  params <- list(
    "feature_pre_filter" = TRUE
    , "is_sparse" = FALSE
  )
  ds <- lgb.Dataset(
    test_data
    , label = test_label
    , params = params
  )
  returned_params <- ds$get_params()
  expect_identical(class(returned_params), "list")
  expect_identical(length(params), length(returned_params))
  expect_identical(sort(names(params)), sort(names(returned_params)))
  for (param_name in names(params)) {
    expect_identical(params[[param_name]], returned_params[[param_name]])
  }
})

test_that("Dataset$get_params() ignores irrelevant parameters", {
  params <- list(
    "feature_pre_filter" = TRUE
    , "is_sparse" = FALSE
    , "nonsense_parameter" = c(1.0, 2.0, 5.0)
  )
  ds <- lgb.Dataset(
    test_data
    , label = test_label
    , params = params
  )
  returned_params <- ds$get_params()
  expect_false("nonsense_parameter" %in% names(returned_params))
})

test_that("Dataset$update_parameters() does nothing for empty inputs", {
  ds <- lgb.Dataset(
    test_data
    , label = test_label
  )
  initial_params <- ds$get_params()
  expect_identical(initial_params, list())

  # update_params() should return "self" so it can be chained
  res <- ds$update_params(
    params = list()
  )
  expect_true(.is_Dataset(res))

  new_params <- ds$get_params()
  expect_identical(new_params, initial_params)
})

test_that("Dataset$update_params() works correctly for recognized Dataset parameters", {
  ds <- lgb.Dataset(
    test_data
    , label = test_label
  )
  initial_params <- ds$get_params()
  expect_identical(initial_params, list())

  new_params <- list(
    "data_random_seed" = 708L
    , "enable_bundle" = FALSE
  )
  res <- ds$update_params(
    params = new_params
  )
  expect_true(.is_Dataset(res))

  updated_params <- ds$get_params()
  for (param_name in names(new_params)) {
    expect_identical(new_params[[param_name]], updated_params[[param_name]])
  }
})

test_that("Dataset's finalizer should not fail on an already-finalized Dataset", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
  )
  expect_true(.is_null_handle(dtest$.__enclos_env__$private$handle))

  dtest$construct()
  expect_false(.is_null_handle(dtest$.__enclos_env__$private$handle))

  dtest$.__enclos_env__$private$finalize()
  expect_true(.is_null_handle(dtest$.__enclos_env__$private$handle))

  # calling finalize() a second time shouldn't cause any issues
  dtest$.__enclos_env__$private$finalize()
  expect_true(.is_null_handle(dtest$.__enclos_env__$private$handle))
})

test_that("lgb.Dataset: should be able to run lgb.train() immediately after using lgb.Dataset() on a file", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
    , params = list(
      verbose = .LGB_VERBOSITY
    )
  )
  tmp_file <- tempfile(pattern = "lgb.Dataset_")
  lgb.Dataset.save(
    dataset = dtest
    , fname = tmp_file
  )

  # read from a local file
  dtest_read_in <- lgb.Dataset(data = tmp_file)

  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )

  # should be able to train right away
  bst <- lgb.train(
    params = param
    , data = dtest_read_in
  )

  expect_true(.is_Booster(x = bst))
})

test_that("lgb.Dataset: should be able to run lgb.cv() immediately after using lgb.Dataset() on a file", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
    , params = list(
      verbosity = .LGB_VERBOSITY
    )
  )
  tmp_file <- tempfile(pattern = "lgb.Dataset_")
  lgb.Dataset.save(
    dataset = dtest
    , fname = tmp_file
  )

  # read from a local file
  dtest_read_in <- lgb.Dataset(data = tmp_file)

  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
    , num_iterations = 5L
    , verbosity = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )

  # should be able to train right away
  bst <- lgb.cv(
    params = param
    , data = dtest_read_in
  )

  expect_true(methods::is(bst, "lgb.CVBooster"))
})

test_that("lgb.Dataset: should be able to be used in lgb.cv() when constructed with categorical feature indices", {
  data("mtcars")
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1L])
  categorical_feature <- which(names(mtcars) %in% c("cyl", "vs", "am", "gear", "carb")) - 1L
  dtrain <- lgb.Dataset(
    data = x
    , label = y
    , categorical_feature = categorical_feature
    , free_raw_data = TRUE
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  # constructing the Dataset frees the raw data
  dtrain$construct()
  params <- list(
    objective = "regression"
    , num_leaves = 2L
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  # cv should reuse the same categorical features without checking the indices
  bst <- lgb.cv(params = params, data = dtrain, stratified = FALSE, nrounds = 1L)
  expect_equal(
    unlist(bst$boosters[[1L]]$booster$params$categorical_feature)
    , categorical_feature - 1L  # 0-based
  )
})


test_that("lgb.Dataset: should be able to use and retrieve long feature names", {
  # set one feature to a value longer than the default buffer size used
  # in LGBM_DatasetGetFeatureNames_R
  feature_names <- names(iris)
  long_name <- strrep("a", 1000L)
  feature_names[1L] <- long_name
  names(iris) <- feature_names
  # check that feature name survived the trip from R to C++ and back
  dtrain <- lgb.Dataset(
    data = as.matrix(iris[, -5L])
    , label = as.numeric(iris$Species) - 1L
  )
  dtrain$construct()
  col_names <- dtrain$get_colnames()
  expect_equal(col_names[1L], long_name)
  expect_equal(nchar(col_names[1L]), 1000L)
})

test_that("lgb.Dataset: should be able to create a Dataset from a text file with a header", {
  train_file <- tempfile(pattern = "train_", fileext = ".csv")
  write.table(
    data.frame(y = rnorm(100L), x1 = rnorm(100L), x2 = rnorm(100L))
    , file = train_file
    , sep = ","
    , col.names = TRUE
    , row.names = FALSE
    , quote = FALSE
  )

  dtrain <- lgb.Dataset(
    data = train_file
    , params = list(
      header = TRUE
      , verbosity = .LGB_VERBOSITY
    )
  )
  dtrain$construct()
  expect_identical(dtrain$get_colnames(), c("x1", "x2"))
  expect_identical(dtrain$get_params(), list(header = TRUE))
  expect_identical(dtrain$dim(), c(100L, 2L))
})

test_that("lgb.Dataset: should be able to create a Dataset from a text file without a header", {
  train_file <- tempfile(pattern = "train_", fileext = ".csv")
  write.table(
    data.frame(y = rnorm(100L), x1 = rnorm(100L), x2 = rnorm(100L))
    , file = train_file
    , sep = ","
    , col.names = FALSE
    , row.names = FALSE
    , quote = FALSE
  )

  dtrain <- lgb.Dataset(
    data = train_file
    , params = list(
      header = FALSE
      , verbosity = .LGB_VERBOSITY
    )
  )
  dtrain$construct()
  expect_identical(dtrain$get_colnames(), c("Column_0", "Column_1"))
  expect_identical(dtrain$get_params(), list(header = FALSE))
  expect_identical(dtrain$dim(), c(100L, 2L))
})

test_that("Dataset: method calls on a Dataset with a null handle should raise an informative error and not segfault", {
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(train$data, label = train$label)
  dtrain$construct()
  dvalid <- dtrain$create_valid(
    data = train$data[seq_len(100L), ]
    , label = train$label[seq_len(100L)]
  )
  dvalid$construct()
  tmp_file <- tempfile(fileext = ".rds")
  saveRDS(dtrain, tmp_file)
  rm(dtrain)
  dtrain <- readRDS(tmp_file)
  expect_error({
    dtrain$construct()
  }, regexp = "Attempting to create a Dataset without any raw data")
  expect_error({
    dtrain$dim()
  }, regexp = "cannot get dimensions before dataset has been constructed")
  expect_error({
    dtrain$get_colnames()
  }, regexp = "cannot get column names before dataset has been constructed")
  expect_error({
    dtrain$get_feature_num_bin(1L)
  }, regexp = "Cannot get number of bins in feature before constructing Dataset.")
  expect_error({
    dtrain$save_binary(fname = tempfile(fileext = ".bin"))
  }, regexp = "Attempting to create a Dataset without any raw data")
  expect_error({
    dtrain$set_categorical_feature(categorical_feature = 1L)
  }, regexp = "cannot set categorical feature after freeing raw data")
  expect_error({
    dtrain$set_reference(reference = dvalid)
  }, regexp = "cannot set reference after freeing raw data")

  tmp_valid_file <- tempfile(fileext = ".rds")
  saveRDS(dvalid, tmp_valid_file)
  rm(dvalid)
  dvalid <- readRDS(tmp_valid_file)
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , free_raw_data = FALSE
  )
  dtrain$construct()
  expect_error({
    dtrain$set_reference(reference = dvalid)
  }, regexp = "cannot get column names before dataset has been constructed")
})

test_that("lgb.Dataset$get_feature_num_bin() works", {
  raw_df <- data.frame(
    all_random = runif(100L)
    , two_vals = rep(c(1.0, 2.0), 50L)
    , three_vals = c(rep(c(0.0, 1.0, 2.0), 33L), 0.0)
    , two_vals_plus_missing = c(rep(c(1.0, 2.0), 49L), NA_real_, NA_real_)
    , all_zero = rep(0.0, 100L)
    , categorical = sample.int(2L, 100L, replace = TRUE)
  )
  n_features <- ncol(raw_df)
  raw_mat <- data.matrix(raw_df)
  min_data_in_bin <- 2L
  ds <- lgb.Dataset(
    raw_mat
    , params = list(min_data_in_bin = min_data_in_bin)
    , categorical_feature = n_features
  )
  ds$construct()
  expected_num_bins <- c(
    100L %/% min_data_in_bin + 1L  # extra bin for zero
    , 3L  # 0, 1, 2
    , 3L  # 0, 1, 2
    , 4L  # 0, 1, 2 + NA
    , 0L  # unused
    , 3L  # 1, 2 + NA
  )
  actual_num_bins <- sapply(1L:n_features, ds$get_feature_num_bin)
  expect_identical(actual_num_bins, expected_num_bins)
  # test using defined feature names
  bins_by_name <- sapply(colnames(raw_mat), ds$get_feature_num_bin)
  expect_identical(unname(bins_by_name), expected_num_bins)
  # test using default feature names
  no_names_mat <- raw_mat
  colnames(no_names_mat) <- NULL
  ds_no_names <- lgb.Dataset(
    no_names_mat
    , params = list(min_data_in_bin = min_data_in_bin)
    , categorical_feature = n_features
  )
  ds_no_names$construct()
  default_names <- lapply(
    X = seq(1L, ncol(raw_mat))
    , FUN = function(i) {
      sprintf("Column_%d", i - 1L)
    }
  )
  bins_by_default_name <- sapply(default_names, ds_no_names$get_feature_num_bin)
  expect_identical(bins_by_default_name, expected_num_bins)
})

test_that("lgb.Dataset can be constructed with categorical features and without colnames", {
  # check that dataset can be constructed
  raw_mat <- matrix(rep(c(0L, 1L), 50L), ncol = 1L)
  ds <- lgb.Dataset(raw_mat, categorical_feature = 1L)$construct()
  sparse_mat <- as(raw_mat, "dgCMatrix")
  ds2 <- lgb.Dataset(sparse_mat, categorical_feature = 1L)$construct()
  # check that the column names are the default ones
  expect_equal(ds$.__enclos_env__$private$colnames, "Column_0")
  expect_equal(ds2$.__enclos_env__$private$colnames, "Column_0")
  # check for error when index is greater than the number of columns
  expect_error({
    lgb.Dataset(raw_mat, categorical_feature = 2L)$construct()
  }, regexp = "supplied a too large value in categorical_feature: 2 but only 1 features")
})

test_that("lgb.Dataset.slice fails with a categorical feature index greater than the number of features", {
  data <- matrix(runif(100L), nrow = 50L, ncol = 2L)
  ds <- lgb.Dataset(data = data, categorical_feature = 3L)
  subset <- ds$slice(1L:20L)
  expect_error({
    subset$construct()
  }, regexp = "supplied a too large value in categorical_feature: 3 but only 2 features")
})
