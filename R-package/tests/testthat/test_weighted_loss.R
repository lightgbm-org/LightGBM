test_that("Gamma regression reacts on 'weight'", {
  n <- 100L
  set.seed(87L)
  X <- matrix(runif(2L * n), ncol = 2L)
  y <- X[, 1L] + X[, 2L] + runif(n)
  X_pred <- X[1L:5L, ]

  params <- list(objective = "gamma", num_threads = .LGB_MAX_THREADS)

  # Unweighted
  dtrain <- lgb.Dataset(X, label = y)
  bst <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 4L
    , verbose = .LGB_VERBOSITY
  )
  pred_unweighted <- predict(bst, X_pred)

  # Constant weight 1
  dtrain <- lgb.Dataset(
    X
    , label = y
    , weight = rep(1.0, n)
  )
  bst <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 4L
    , verbose = .LGB_VERBOSITY
  )
  pred_weighted_1 <- predict(bst, X_pred)

  # Constant weight 2
  dtrain <- lgb.Dataset(
    X
    , label = y
    , weight = rep(2.0, n)
  )
  bst <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 4L
    , verbose = .LGB_VERBOSITY
  )
  pred_weighted_2 <- predict(bst, X_pred)

  # Non-constant weights
  dtrain <- lgb.Dataset(
    X
    , label = y
    , weight = seq(0.0, 1.0, length.out = n)
  )
  bst <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = 4L
    , verbose = .LGB_VERBOSITY
  )
  pred_weighted <- predict(bst, X_pred)

  expect_equal(pred_unweighted, pred_weighted_1)
  expect_equal(pred_weighted_1, pred_weighted_2)
  expect_false(all(pred_unweighted == pred_weighted))
})
