NROUNDS <- 10L
MAX_DEPTH <- 3L
N <- nrow(iris)
X <- data.matrix(iris[2L:4L])
FEAT <- colnames(X)
NCLASS <- nlevels(iris[, 5L])

model_reg <- lgb.train(
  params = list(
    objective = "regression"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
  )
  , data = lgb.Dataset(X, label = iris[, 1L])
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_binary <- lgb.train(
  params = list(
    objective = "binary"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
  )
  , data = lgb.Dataset(X, label = iris[, 5L] == "setosa")
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_multiclass <- lgb.train(
  params = list(
    objective = "multiclass"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
    , num_classes = NCLASS
  )
  , data = lgb.Dataset(X, label = as.integer(iris[, 5L]) - 1L)
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_rank <- lgb.train(
  params = list(
    objective = "lambdarank"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
    , lambdarank_truncation_level = 3L
  )
  , data = lgb.Dataset(
    X
    , label = as.integer(iris[, 1L] > 5.8)
    , group = rep(10L, times = 15L)
  )
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

models <- list(
  reg = model_reg
  , bin = model_binary
  , multi = model_multiclass
  , rank = model_rank
)

for (model_name in names(models)) {
  model <- models[[model_name]]
  expected_n_trees <- NROUNDS
  if (model_name == "multi") {
    expected_n_trees <- NROUNDS * NCLASS
  }
  df <- as.data.frame(lgb.model.dt.tree(model))
  df_list <- split(df, f = df$tree_index, drop = TRUE)

  df_leaf <- df[!is.na(df$leaf_index), ]
  df_internal <- df[is.na(df$leaf_index), ]

  test_that("lgb.model.dt.tree() returns the right number of trees", {
    expect_equal(length(unique(df$tree_index)), expected_n_trees)
  })

  test_that("num_iteration can return less trees", {
    expect_equal(
      length(unique(lgb.model.dt.tree(model, num_iteration = 2L)$tree_index))
      , 2L * (if (model_name == "multi") NCLASS else 1L)
    )
  })

  test_that("Tree index from lgb.model.dt.tree() is in 0:(NROUNS-1)", {
    expect_equal(unique(df$tree_index), (0L:(expected_n_trees - 1L)))
  })

  test_that("Depth calculated from lgb.model.dt.tree() respects max.depth", {
    expect_true(max(df$depth) <= MAX_DEPTH)
  })

  test_that("Each tree from lgb.model.dt.tree() has single root node", {
    expect_equal(
      unname(sapply(df_list, function(df) sum(df$depth == 0L)))
      , rep(1L, expected_n_trees)
    )
  })

  test_that("Each tree from lgb.model.dt.tree() has two depth 1 nodes", {
    expect_equal(
      unname(sapply(df_list, function(df) sum(df$depth == 1L)))
      , rep(2L, expected_n_trees)
    )
  })

  test_that("leaves from lgb.model.dt.tree() do not have split info", {
    internal_node_cols <- c(
      "split_index"
      , "split_feature"
      , "split_gain"
      , "threshold"
      , "decision_type"
      , "default_left"
      , "internal_value"
      , "internal_count"
    )
    expect_true(all(is.na(df_leaf[internal_node_cols])))
  })

  test_that("leaves from lgb.model.dt.tree() have valid leaf info", {
    expect_true(all(df_leaf$leaf_index %in% 0L:(2.0^MAX_DEPTH - 1.0)))
    expect_true(all(is.finite(df_leaf$leaf_value)))
    expect_true(all(df_leaf$leaf_count > 0L & df_leaf$leaf_count <= N))
  })

  test_that("non-leaves from lgb.model.dt.tree() do not have leaf info", {
    leaf_node_cols <- c(
      "leaf_index", "leaf_parent", "leaf_value", "leaf_count"
    )
    expect_true(all(is.na(df_internal[leaf_node_cols])))
  })

  test_that("non-leaves from lgb.model.dt.tree() have valid split info", {
    expect_true(
      all(
        sapply(
          split(df_internal, df_internal$tree_index),
          function(x) all(x$split_index %in% 0L:(nrow(x) - 1L))
        )
      )
    )

    expect_true(all(df_internal$split_feature %in% FEAT))

    num_cols <- c("split_gain", "threshold", "internal_value")
    expect_true(all(is.finite(unlist(df_internal[, num_cols]))))

    # range of decision type?
    expect_true(all(df_internal$default_left %in% c(TRUE, FALSE)))

    counts <- df_internal$internal_count
    expect_true(all(counts > 1L & counts <= N))
  })
}

test_that("num_iteration and start_iteration work as expected", {
  set.seed(1L)
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  bst <- lightgbm(
    data = as.matrix(train$data)
    , label = train$label
    , params = list(objective = "binary", num_threads = .LGB_MAX_THREADS)
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
  )

  first2 <- lgb.model.dt.tree(bst, num_iteration = 2L)
  last3 <- lgb.model.dt.tree(bst, num_iteration = 3L, start_iteration = 3L)
  all5 <- lgb.model.dt.tree(bst)
  too_many <- lgb.model.dt.tree(bst, num_iteration = 10L)

  expect_equal(data.table::rbindlist(list(first2, last3)), all5)
  expect_equal(too_many, all5)

  # Check tree indices
  expect_equal(unique(first2[["tree_index"]]), 0L:1L)
  expect_equal(unique(last3[["tree_index"]]), 2L:4L)
  expect_equal(unique(all5[["tree_index"]]), 0L:4L)
})
