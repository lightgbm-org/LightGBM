data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset(agaricus.test$data, label = agaricus.test$label)
watchlist <- list(eval = dtest, train = dtrain)

logregobj <- function(preds, dtrain) {
  labels <- get_field(dtrain, "label")
  preds <- 1.0 / (1.0 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1.0 - preds)
  return(list(grad = grad, hess = hess))
}

# User-defined evaluation function returns a pair (metric_name, result, higher_better)
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# Keep this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- get_field(dtrain, "label")
  preds <- 1.0 / (1.0 + exp(-preds))
  err <- as.numeric(sum(labels != (preds > 0.5))) / length(labels)
  return(list(
    name = "error"
    , value = err
    , higher_better = FALSE
  ))
}

param <- list(
  num_leaves = 8L
  , learning_rate = 1.0
  , objective = logregobj
  , metric = "auc"
  , verbose = .LGB_VERBOSITY
  , num_threads = .LGB_MAX_THREADS
)
num_round <- 10L

test_that("custom objective works", {
  bst <- lgb.train(param, dtrain, num_round, watchlist, eval = evalerror)
  expect_false(is.null(bst$record_evals))
})

test_that("using a custom objective, custom eval, and no other metrics works", {
  set.seed(708L)
  bst <- lgb.train(
    params = list(
      num_leaves = 8L
      , learning_rate = 1.0
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = 4L
    , valids = watchlist
    , obj = logregobj
    , eval = evalerror
  )
  expect_false(is.null(bst$record_evals))
  expect_equal(bst$best_iter, 4L)
  expect_true(abs(bst$best_score - 0.000621) < .LGB_NUMERIC_TOLERANCE)

  eval_results <- bst$eval_valid(feval = evalerror)[[1L]]
  expect_true(eval_results[["data_name"]] == "eval")
  expect_true(abs(eval_results[["value"]] - 0.0006207325) < .LGB_NUMERIC_TOLERANCE)
  expect_true(eval_results[["name"]] == "error")
  expect_false(eval_results[["higher_better"]])
})

test_that("using a custom objective that returns wrong shape grad or hess raises an informative error", {
  bad_grad <- function(preds, dtrain) {
    return(list(grad = numeric(0L), hess = rep(1.0, length(preds))))
  }
  bad_hess <- function(preds, dtrain) {
    return(list(grad = rep(1.0, length(preds)), hess = numeric(0L)))
  }
  params <- list(num_leaves = 3L, verbose = .LGB_VERBOSITY)
  expect_error({
    lgb.train(params = params, data = dtrain, obj = bad_grad)
  }, sprintf("Expected custom objective function to return grad with length %d, got 0.", nrow(dtrain)))
  expect_error({
    lgb.train(params = params, data = dtrain, obj = bad_hess)
  }, sprintf("Expected custom objective function to return hess with length %d, got 0.", nrow(dtrain)))
})
