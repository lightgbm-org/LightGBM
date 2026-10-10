#' @name lgb.train
#' @title Main training logic for LightGBM
#' @description Low-level R interface to train a LightGBM model. Unlike \code{\link{lightgbm}},
#'              this function is focused on performance (e.g. speed, memory efficiency). It is also
#'              less likely to have breaking API changes in new releases than \code{\link{lightgbm}}.
#' @inheritParams lgb_shared_params
#' @param valids a list of \code{lgb.Dataset} objects, used for validation
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the
#'                   booster model into a predictor model which frees up memory and the
#'                   original datasets
#' @inheritSection lgb_shared_params Early Stopping
#' @return a trained booster model \code{lgb.Booster}.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , early_stopping_rounds = 3L
#' )
#' }
#'
#' @export
lgb.train <- function(params = list(),
                      data,
                      nrounds = 100L,
                      valids = list(),
                      obj = NULL,
                      eval = NULL,
                      verbose = 1L,
                      record = TRUE,
                      eval_freq = 1L,
                      init_model = NULL,
                      early_stopping_rounds = NULL,
                      callbacks = list(),
                      reset_data = FALSE,
                      serializable = TRUE) {

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }
  if (!.is_Dataset(x = data)) {
    stop("lgb.train: data must be an lgb.Dataset instance")
  }
  if (length(valids) > 0L) {
    if (!identical(class(valids), "list") || !all(vapply(valids, .is_Dataset, logical(1L)))) {
      stop("lgb.train: valids must be a list of lgb.Dataset elements")
    }
    evnames <- names(valids)
    if (is.null(evnames) || !all(nzchar(evnames))) {
      stop("lgb.train: each element of valids must have a name")
    }
  }

  # set some parameters, resolving the way they were passed in with other parameters
  # in `params`.
  # this ensures that the model stored with Booster$save() correctly represents
  # what was passed in
  params <- .check_wrapper_param(
    main_param_name = "verbosity"
    , params = params
    , alternative_kwarg_value = verbose
  )
  params <- .check_wrapper_param(
    main_param_name = "num_iterations"
    , params = params
    , alternative_kwarg_value = nrounds
  )
  params <- .check_wrapper_param(
    main_param_name = "metric"
    , params = params
    , alternative_kwarg_value = NULL
  )
  params <- .check_wrapper_param(
    main_param_name = "objective"
    , params = params
    , alternative_kwarg_value = obj
  )
  params <- .check_wrapper_param(
    main_param_name = "early_stopping_round"
    , params = params
    , alternative_kwarg_value = early_stopping_rounds
  )
  early_stopping_rounds <- params[["early_stopping_round"]]

  # extract any function objects passed for objective or metric
  fobj <- NULL
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "none"
  }

  # If eval is a single function, store it as a 1-element list
  # (for backwards compatibility). If it is a list of functions, store
  # all of them. This makes it possible to pass any mix of strings like "auc"
  # and custom functions to eval
  params <- .check_eval(params = params, eval = eval)
  eval_functions <- list(NULL)
  if (is.function(eval)) {
    eval_functions <- list(eval)
  }
  if (methods::is(eval, "list")) {
    eval_functions <- Filter(
      f = is.function
      , x = eval
    )
  }

  # Init predictor to empty
  predictor <- NULL

  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(modelfile = init_model)
  } else if (.is_Booster(x = init_model)) {
    predictor <- init_model$to_predictor()
  }

  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1L
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1L
  }
  end_iteration <- begin_iteration + params[["num_iterations"]] - 1L

  # pop interaction_constraints off of params. It needs some preprocessing on the
  # R side before being passed into the Dataset object
  interaction_constraints <- params[["interaction_constraints"]]
  params["interaction_constraints"] <- NULL

  # Construct datasets, if needed
  data$update_params(params = params)
  data$construct()

  # Check interaction constraints
  params[["interaction_constraints"]] <- .check_interaction_constraints(
    interaction_constraints = interaction_constraints
    , column_names = data$get_colnames()
  )

  # Update parameters with parsed parameters
  data$update_params(params)

  # Create the predictor set
  data$.__enclos_env__$private$set_predictor(predictor)

  valid_contain_train <- FALSE
  train_data_name <- "train"
  reduced_valid_sets <- list()

  # Parse validation datasets
  if (length(valids) > 0L) {

    for (key in names(valids)) {

      # Use names to get validation datasets
      valid_data <- valids[[key]]

      # Check for duplicate train/validation dataset
      if (identical(data, valid_data)) {
        valid_contain_train <- TRUE
        train_data_name <- key
        next
      }

      # Update parameters, data
      valid_data$update_params(params)
      valid_data$set_reference(data)
      reduced_valid_sets[[key]] <- valid_data

    }

  }

  # Add printing log callback
  if (params[["verbosity"]] > 0L && eval_freq > 0L) {
    callbacks <- .add_cb(
        cb_list = callbacks
        , cb = cb_print_evaluation(period = eval_freq)
    )
  }

  # Add evaluation log callback
  if (record && length(valids) > 0L) {
    callbacks <- .add_cb(
        cb_list = callbacks
        , cb = cb_record_evaluation()
    )
  }

  # Did user pass parameters that indicate they want to use early stopping?
  using_early_stopping <- !is.null(early_stopping_rounds) && early_stopping_rounds > 0L

  boosting_param_names <- .PARAMETER_ALIASES()[["boosting"]]
  using_dart <- any(
    sapply(
      X = boosting_param_names
      , FUN = function(param) {
        identical(params[[param]], "dart")
      }
    )
  )

  # Cannot use early stopping with 'dart' boosting
  if (using_dart) {
    if (using_early_stopping) {
      warning("Early stopping is not available in 'dart' mode.")
    }
    using_early_stopping <- FALSE

    # Remove the cb_early_stop() function if it was passed in to callbacks
    callbacks <- Filter(
      f = function(cb_func) {
        !identical(attr(cb_func, "name"), "cb_early_stop")
      }
      , x = callbacks
    )
  }

  # If user supplied early_stopping_rounds, add the early stopping callback
  if (using_early_stopping) {
    callbacks <- .add_cb(
      cb_list = callbacks
      , cb = cb_early_stop(
        stopping_rounds = early_stopping_rounds
        , first_metric_only = isTRUE(params[["first_metric_only"]])
        , verbose = params[["verbosity"]] > 0L
      )
    )
  }

  cb <- .categorize_callbacks(cb_list = callbacks)

  # Construct booster with datasets
  booster <- Booster$new(params = params, train_set = data)
  if (valid_contain_train) {
    booster$set_train_data_name(name = train_data_name)
  }

  for (key in names(reduced_valid_sets)) {
    booster$add_valid(data = reduced_valid_sets[[key]], name = key)
  }

  # Callback env
  env <- CB_ENV$new()
  env$model <- booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  # Start training model using number of iterations to start and end with
  for (i in seq.int(from = begin_iteration, to = end_iteration)) {

    # Overwrite iteration in environment
    env$iteration <- i
    env$eval_list <- list()

    # Loop through "pre_iter" element
    for (f in cb$pre_iter) {
      f(env)
    }

    # Update one boosting iteration
    booster$update(fobj = fobj)

    # Prepare collection of evaluation results
    eval_list <- list()

    # Collection: Has validation dataset?
    if (length(valids) > 0L) {

      # Get evaluation results with passed-in functions
      for (eval_function in eval_functions) {

        # Validation has training dataset?
        if (valid_contain_train) {
          eval_list <- append(eval_list, booster$eval_train(feval = eval_function))
        }

        eval_list <- append(eval_list, booster$eval_valid(feval = eval_function))
      }

      # Calling booster$eval_valid() will get
      # evaluation results with the metrics in params$metric by calling LGBM_BoosterGetEval_R",
      # so need to be sure that gets called, which it wouldn't be above if no functions
      # were passed in
      if (length(eval_functions) == 0L) {
        if (valid_contain_train) {
          eval_list <- append(eval_list, booster$eval_train(feval = eval_function))
        }
        eval_list <- append(eval_list, booster$eval_valid(feval = eval_function))
      }

    }

    # Write evaluation result in environment
    env$eval_list <- eval_list

    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }

    # Check for early stopping and break if needed
    if (env$met_early_stop) break

  }

  # check if any valids were given other than the training data
  non_train_valid_names <- names(valids)[!(names(valids) == train_data_name)]
  first_valid_name <- non_train_valid_names[1L]

  # When early stopping is not activated, we compute the best iteration / score ourselves by
  # selecting the first metric and the first dataset
  if (record && length(non_train_valid_names) > 0L && is.na(env$best_score)) {

    # when using a custom eval function, the metric name is returned from the
    # function, so figure it out from record_evals
    if (!is.null(eval_functions[1L])) {
      first_metric <- names(booster$record_evals[[first_valid_name]])[1L]
    } else {
      first_metric <- booster$.__enclos_env__$private$eval_names[1L]
    }

    .find_best <- which.min
    if (isTRUE(env$eval_list[[1L]]$higher_better[1L])) {
      .find_best <- which.max
    }
    booster$best_iter <- unname(
      .find_best(
        unlist(
          booster$record_evals[[first_valid_name]][[first_metric]][[.EVAL_KEY()]]
        )
      )
    )
    booster$best_score <- booster$record_evals[[first_valid_name]][[first_metric]][[.EVAL_KEY()]][[booster$best_iter]]
  }

  # Check for booster model conversion to predictor model
  if (reset_data) {

    # Store temporarily model data elsewhere
    booster_old <- list(
      best_iter = booster$best_iter
      , best_score = booster$best_score
      , record_evals = booster$record_evals
    )

    # Reload model
    booster <- lgb.load(model_str = booster$save_model_to_string())
    booster$best_iter <- booster_old$best_iter
    booster$best_score <- booster_old$best_score
    booster$record_evals <- booster_old$record_evals

  }

  if (serializable) {
    booster$save_raw()
  }

  return(booster)

}
