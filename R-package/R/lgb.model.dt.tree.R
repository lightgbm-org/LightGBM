#' @name lgb.model.dt.tree
#' @title Parse a LightGBM model json dump
#' @description Parse a LightGBM model json dump into a \code{data.table} structure.
#' @param model object of class \code{lgb.Booster}.
#' @param num_iteration Number of iterations to include. NULL or <= 0 means use best iteration.
#' @param start_iteration Index (1-based) of the first boosting round to include in the output.
#'        For example, passing \code{start_iteration=5, num_iteration=3} for a regression model
#'        means "return information about the fifth, sixth, and seventh trees".
#'
#'        \emph{New in version 4.4.0}
#'
#' @return
#' A \code{data.table} with detailed information about model trees' nodes and leaves.
#'
#' The columns of the \code{data.table} are:
#'
#' \itemize{
#'  \item{\code{tree_index}: ID of a tree in a model (integer)}
#'  \item{\code{split_index}: ID of a node in a tree (integer)}
#'  \item{\code{split_feature}: for a node, it's a feature name (character);
#'                              for a leaf, it simply labels it as \code{"NA"}}
#'  \item{\code{node_parent}: ID of the parent node for current node (integer)}
#'  \item{\code{leaf_index}: ID of a leaf in a tree (integer)}
#'  \item{\code{leaf_parent}: ID of the parent node for current leaf (integer)}
#'  \item{\code{split_gain}: Split gain of a node}
#'  \item{\code{threshold}: Splitting threshold value of a node}
#'  \item{\code{decision_type}: Decision type of a node}
#'  \item{\code{default_left}: Determine how to handle NA value, TRUE -> Left, FALSE -> Right}
#'  \item{\code{internal_value}: Node value}
#'  \item{\code{internal_count}: The number of observation collected by a node}
#'  \item{\code{leaf_value}: Leaf value}
#'  \item{\code{leaf_count}: The number of observation collected by a leaf}
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
#'   , learning_rate = 0.01
#'   , num_leaves = 63L
#'   , max_depth = -1L
#'   , min_data_in_leaf = 1L
#'   , min_sum_hessian_in_leaf = 1.0
#'   , num_threads = 2L
#' )
#' model <- lgb.train(params, dtrain, 10L)
#'
#' tree_dt <- lgb.model.dt.tree(model)
#' }
#' @importFrom data.table := rbindlist
#' @importFrom jsonlite fromJSON
#' @export
lgb.model.dt.tree <- function(
    model, num_iteration = NULL, start_iteration = 1L
  ) {

  json_model <- lgb.dump(
    booster = model
    , num_iteration = num_iteration
    , start_iteration = start_iteration
  )

  parsed_json_model <- jsonlite::fromJSON(
    txt = json_model
    , simplifyVector = TRUE
    , simplifyDataFrame = FALSE
    , simplifyMatrix = FALSE
    , flatten = FALSE
  )

  # Parse tree model
  tree_list <- lapply(
    X = parsed_json_model$tree_info
    , FUN = .single_tree_parse
  )

  # Combine into single data.table
  tree_dt <- data.table::rbindlist(l = tree_list, use.names = TRUE)

  # Substitute feature index with the actual feature name

  # Since the index comes from C++ (which is 0-indexed), be sure
  # to add 1 (e.g. index 28 means the 29th feature in feature_names)
  split_feature_indx <- tree_dt[, split_feature] + 1L

  # Get corresponding feature names. Positions in split_feature_indx
  # which are NA will result in an NA feature name
  feature_names <- parsed_json_model$feature_names[split_feature_indx]
  tree_dt[, split_feature := feature_names]

  return(tree_dt)
}


#' @importFrom data.table := data.table rbindlist
.single_tree_parse <- function(lgb_tree) {
  tree_info_cols <- c(
    "split_index"
    , "split_feature"
    , "split_gain"
    , "threshold"
    , "decision_type"
    , "default_left"
    , "internal_value"
    , "internal_count"
  )

  # Traverse tree function
  pre_order_traversal <- function(env = NULL, tree_node_leaf, current_depth = 0L, parent_index = NA_integer_) {

    if (is.null(env)) {
      # Setup initial default data.table with default types
      env <- new.env(parent = emptyenv())
      env$single_tree_dt <- list()
      env$single_tree_dt[[1L]] <- data.table::data.table(
        tree_index = integer(0L)
        , depth = integer(0L)
        , split_index = integer(0L)
        , split_feature = integer(0L)
        , node_parent = integer(0L)
        , leaf_index = integer(0L)
        , leaf_parent = integer(0L)
        , split_gain = numeric(0L)
        , threshold = numeric(0L)
        , decision_type = character(0L)
        , default_left = character(0L)
        , internal_value = integer(0L)
        , internal_count = integer(0L)
        , leaf_value = integer(0L)
        , leaf_count = integer(0L)
      )
      # start tree traversal
      pre_order_traversal(
        env = env
        , tree_node_leaf = tree_node_leaf
        , current_depth = current_depth
        , parent_index = parent_index
      )
    } else {

      # Check if split index is not null in leaf
      if (!is.null(tree_node_leaf$split_index)) {

        # update data.table
        env$single_tree_dt[[length(env$single_tree_dt) + 1L]] <- c(
          tree_node_leaf[tree_info_cols]
          , list("depth" = current_depth, "node_parent" = parent_index)
        )

        # Traverse tree again both left and right
        pre_order_traversal(
          env = env
          , tree_node_leaf = tree_node_leaf$left_child
          , current_depth = current_depth + 1L
          , parent_index = tree_node_leaf$split_index
        )
        pre_order_traversal(
          env = env
          , tree_node_leaf = tree_node_leaf$right_child
          , current_depth = current_depth + 1L
          , parent_index = tree_node_leaf$split_index
        )
      } else if (!is.null(tree_node_leaf$leaf_index)) {

        # update list
        env$single_tree_dt[[length(env$single_tree_dt) + 1L]] <- c(
          tree_node_leaf[c("leaf_index", "leaf_value", "leaf_count")]
          , list("depth" = current_depth, "leaf_parent" = parent_index)
        )
      }
    }
    return(env$single_tree_dt)
  }

  # Traverse structure and rowbind everything
  single_tree_dt <- data.table::rbindlist(
    pre_order_traversal(tree_node_leaf = lgb_tree$tree_structure)
    , use.names = TRUE
    , fill = TRUE
  )

  # Store index
  single_tree_dt[, tree_index := lgb_tree$tree_index]

  return(single_tree_dt)
}
