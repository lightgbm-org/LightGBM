#' @name lgb.importance
#' @title Compute feature importance in a model
#' @description Creates a \code{data.table} of feature importances in a model.
#' @param model object of class \code{lgb.Booster}.
#' @param percentage whether to show importance in relative percentage.
#'
#' @return For a tree model, a \code{data.table} with the following columns:
#' \itemize{
#'   \item{\code{Feature}: Feature names in the model.}
#'   \item{\code{Gain}: The total gain of this feature's splits.}
#'   \item{\code{Cover}: The number of observation related to this feature.}
#'   \item{\code{Frequency}: The number of times a feature split in trees.}
#' }
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' params <- list(
#'   objective = "binary"
#'   , learning_rate = 0.1
#'   , max_depth = -1L
#'   , min_data_in_leaf = 1L
#'   , min_sum_hessian_in_leaf = 1.0
#'   , num_threads = 2L
#' )
#' model <- lgb.train(
#'     params = params
#'     , data = dtrain
#'     , nrounds = 5L
#' )
#'
#' tree_imp1 <- lgb.importance(model, percentage = TRUE)
#' tree_imp2 <- lgb.importance(model, percentage = FALSE)
#' }
#' @importFrom data.table := setnames setorderv
#' @export
lgb.importance <- function(model, percentage = TRUE) {

  if (!.is_Booster(x = model)) {
    stop("'model' has to be an object of class lgb.Booster")
  }

  # Setup importance
  tree_dt <- lgb.model.dt.tree(model = model)

  # Extract elements
  tree_imp_dt <- tree_dt[
    !is.na(split_index)
    , .(Gain = sum(split_gain), Cover = sum(internal_count), Frequency = .N)
    , by = "split_feature"
  ]

  data.table::setnames(
    x = tree_imp_dt
    , old = "split_feature"
    , new = "Feature"
  )

  # Sort features by Gain
  data.table::setorderv(
    x = tree_imp_dt
    , cols = "Gain"
    , order = -1L
  )

  # Check if relative values are requested
  if (percentage) {
    tree_imp_dt[, `:=`(
      Gain = Gain / sum(Gain)
      , Cover = Cover / sum(Cover)
      , Frequency = Frequency / sum(Frequency)
    )]
  }

  # adding an empty [] to ensure the table is printed the first time print.data.table() is called
  return(tree_imp_dt[])

}
