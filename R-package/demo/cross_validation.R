library(lightgbm)

# load in the agaricus dataset
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset.create.valid(dtrain, data = agaricus.test$data, label = agaricus.test$label)

nrounds <- 2L
param <- list(
  num_leaves = 4L
  , learning_rate = 1.0
  , objective = "binary"
)

print("Running cross validation")
# Do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
lgb.cv(
  param
  , dtrain
  , nrounds
  , nfold = 5L
  , eval = "binary_error"
)

print("Running cross validation, disable standard deviation display")
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
lgb.cv(
  param
  , dtrain
  , nrounds
  , nfold = 5L
  , eval = "binary_error"
  , showsd = FALSE
)

# You can also do cross validation with customized loss function
print("Running cross validation, with cutomsized loss function")

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
# For example, we are doing logistic loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- get_field(dtrain, "label")
  preds <- 1.0 / (1.0 + exp(-preds))
  err <- as.numeric(sum(labels != (preds > 0.5))) / length(labels)
  return(list(name = "error", value = err, higher_better = FALSE))
}

# train with customized objective
lgb.cv(
  params = param
  , data = dtrain
  , nrounds = nrounds
  , obj = logregobj
  , eval = evalerror
  , nfold = 5L
)
