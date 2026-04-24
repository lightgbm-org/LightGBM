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

models <- list(
  reg = model_reg
)

for (model_name in names(models)){
  model <- models[[model_name]]
  modelDT <- lgb.model.dt.tree(model)

  test_that("lgb.plot.tree fails when a non existing tree is selected", {
    expect_error({
      lgb.plot.tree(model, -1L)
    }, regexp = paste0(
      "lgb.plot.tree: All values of 'tree' should be between 0 and the total number of trees in the model minus one"))
  })
  test_that("lgb.plot.tree fails when a non existing tree is selected", {
    expect_error({
      lgb.plot.tree(model, 999L)
    }, regexp = paste0(
      "lgb.plot.tree: All values of 'tree' should be between 0 and the total number of trees in the model minus one"))
  })
  test_that("lgb.plot.tree fails when a non numeric tree is selected", {
    expect_error({
      lgb.plot.tree(model, "a")
    }, regexp = "lgb.plot.tree: 'tree' must only contain integers.")
  })
  test_that("lgb.plot.tree fails when a non integer tree is selected", {
    expect_error({
      lgb.plot.tree(model, 1.5)
    }, regexp = "lgb.plot.tree: 'tree' must only contain integers.")
  })
  test_that("lgb.plot.tree fails when a non lgb.Booster model is passed", {
    expect_error({
      lgb.plot.tree(1L, 0L)
    }, regexp = paste0("lgb.plot.tree: model should be an ", sQuote("lgb.Booster")))
  })
}
