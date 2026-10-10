library(lightgbm)

# Load in the agaricus dataset
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")

dtrain <- lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
dtest <- lgb.Dataset.create.valid(dtrain, data = agaricus.test$data, label = agaricus.test$label)

# Note: for customized objective function, we leave objective as default
# Note: what we are getting is margin value in prediction
# You must know what you are doing
param <- list(
  num_leaves = 4L
  , learning_rate = 1.0
)
valids <- list(eval = dtest)
num_round <- 20L

# User define objective function, given prediction, return gradient and second order gradient
# This is loglikelihood loss
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
# The built-in evaluation error assumes input is after logistic transformation
# Keep this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- get_field(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0.5))) / length(labels)
  return(list(name = "error", value = err, higher_better = FALSE))
}
print("Start training with early Stopping setting")

bst <- lgb.train(
  param
  , dtrain
  , num_round
  , valids
  , obj = logregobj
  , eval = evalerror
  , early_stopping_round = 3L
)
