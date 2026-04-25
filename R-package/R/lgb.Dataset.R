#' @name lgb_shared_dataset_params
#' @title Shared Dataset parameter docs
#' @description Parameter docs for fields used in \code{lgb.Dataset} construction
#' @param label vector of labels to use as the target variable
#' @param weight numeric vector of sample weights
#' @param init_score initial score is the base prediction lightgbm will boost from
#' @param group used for learning-to-rank tasks. An integer vector describing how to
#'              group rows together as ordered results from the same set of candidate results
#'              to be ranked. For example, if you have a 100-document dataset with
#'              \code{group = c(10, 20, 40, 10, 10, 10)}, that means that you have 6 groups,
#'              where the first 10 records are in the first group, records 11-30 are in the
#'              second group, etc.
#' @details This page contains shared documentation for dataset-related parameters used throughout the package.
#' @keywords internal
NULL

# [description] List of valid keys for "info" arguments in lgb.Dataset.
#               Wrapped in a function to take advantage of lazy evaluation
#               (so it doesn't matter what order R sources files during installation).
# [return] A character vector of names.
.INFO_KEYS <- function() {
  return(c("label", "weight", "init_score", "group"))
}

#' @importFrom methods is
#' @importFrom R6 R6Class
#' @importFrom utils modifyList
Dataset <- R6::R6Class(

  classname = "lgb.Dataset",
  cloneable = FALSE,
  public = list(

    # Initialize will create a starter dataset
    initialize = function(data,
                          params = list(),
                          reference = NULL,
                          colnames = NULL,
                          categorical_feature = NULL,
                          predictor = NULL,
                          free_raw_data = TRUE,
                          used_indices = NULL,
                          label = NULL,
                          weight = NULL,
                          group = NULL,
                          init_score = NULL) {

      # validate inputs early to avoid unnecessary computation
      if (!(is.null(reference) || .is_Dataset(reference))) {
          stop("lgb.Dataset: If provided, reference must be a ", sQuote("lgb.Dataset", q = FALSE))
      }
      if (!(is.null(predictor) || .is_Predictor(predictor))) {
          stop("lgb.Dataset: If provided, predictor must be a ", sQuote("lgb.Predictor", q = FALSE))
      }

      info <- list()
      if (!is.null(label)) {
        info[["label"]] <- label
      }
      if (!is.null(weight)) {
        info[["weight"]] <- weight
      }
      if (!is.null(group)) {
        info[["group"]] <- group
      }
      if (!is.null(init_score)) {
        info[["init_score"]] <- init_score
      }

      # Check for matrix format
      if (is.matrix(data)) {
        # Check whether matrix is the correct type first ("double")
        if (storage.mode(data) != "double") {
          storage.mode(data) <- "double"
        }
      }

      # Setup private attributes
      private$raw_data <- data
      private$params <- params
      private$reference <- reference
      private$colnames <- colnames

      private$categorical_feature <- categorical_feature
      private$predictor <- predictor
      private$free_raw_data <- free_raw_data
      private$used_indices <- sort(used_indices, decreasing = FALSE)
      private$info <- info
      private$version <- 0L

      return(invisible(NULL))

    },

    create_valid = function(data,
                            label = NULL,
                            weight = NULL,
                            group = NULL,
                            init_score = NULL,
                            params = list()) {

      # the Dataset's existing parameters should be overwritten by any passed in to this call
      params <- modifyList(private$params, params)

      # Create new dataset
      ret <- Dataset$new(
        data = data
        , params = params
        , reference = self
        , colnames = private$colnames
        , categorical_feature = private$categorical_feature
        , predictor = private$predictor
        , free_raw_data = private$free_raw_data
        , used_indices = NULL
        , label = label
        , weight = weight
        , group = group
        , init_score = init_score
      )

      return(invisible(ret))

    },

    # Dataset constructor
    construct = function() {

      # Check for handle null
      if (!.is_null_handle(x = private$handle)) {
        return(invisible(self))
      }

      # Get feature names
      cnames <- NULL
      if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {
        cnames <- colnames(private$raw_data)
      }

      # set feature names if they do not exist
      if (is.null(private$colnames) && !is.null(cnames)) {
        private$colnames <- as.character(cnames)
      }

      # Get categorical feature index
      if (!is.null(private$categorical_feature)) {

        # Check for character name
        if (is.character(private$categorical_feature)) {

            cate_indices <- as.list(match(private$categorical_feature, private$colnames) - 1L)

            # Provided indices, but some indices are missing?
            if (anyNA(cate_indices)) {
              stop(
                "lgb.Dataset.construct: supplied an unknown feature in categorical_feature: "
                , sQuote(private$categorical_feature[is.na(cate_indices)], q = FALSE)
              )
            }

          } else {

            # Check if more categorical features were output over the feature space
            data_is_not_filename <- !is.character(private$raw_data)
            if (
              data_is_not_filename
              && !is.null(private$raw_data)
              && is.null(private$used_indices)
              && max(private$categorical_feature) > ncol(private$raw_data)
            ) {
              stop(
                "lgb.Dataset.construct: supplied a too large value in categorical_feature: "
                , max(private$categorical_feature)
                , " but only "
                , ncol(private$raw_data)
                , " features"
              )
            }

            # Store indices as [0, n-1] indexed instead of [1, n] indexed
            cate_indices <- as.list(private$categorical_feature - 1L)

          }

        # Store indices for categorical features
        private$params$categorical_feature <- cate_indices

      }

      # Generate parameter str
      params_str <- .params2str(params = private$params)

      # Get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(private$reference)) {
        ref_handle <- private$reference$.__enclos_env__$private$get_handle()
      }

      # not subsetting, constructing from raw data
      if (is.null(private$used_indices)) {

        if (is.null(private$raw_data)) {
          stop(paste0(
            "Attempting to create a Dataset without any raw data. "
            , "This can happen if the Dataset's finalizer was called or if this Dataset was saved with saveRDS(). "
            , "To avoid this error in the future, use lgb.Dataset.save() or "
            , "Dataset$save_binary() to save lightgbm Datasets."
          ))
        }

        # Are we using a data file?
        if (is.character(private$raw_data)) {

          handle <- .Call(
            LGBM_DatasetCreateFromFile_R
            , path.expand(private$raw_data)
            , params_str
            , ref_handle
          )

        } else if (is.matrix(private$raw_data)) {

          # Are we using a matrix?
          handle <- .Call(
            LGBM_DatasetCreateFromMat_R
            , private$raw_data
            , nrow(private$raw_data)
            , ncol(private$raw_data)
            , params_str
            , ref_handle
          )

        } else if (methods::is(private$raw_data, "dgCMatrix")) {
          if (length(private$raw_data@p) > 2147483647L) {
            stop("Cannot support large CSC matrix")
          }
          # Are we using a dgCMatrix (sparse matrix column compressed)
          handle <- .Call(
            LGBM_DatasetCreateFromCSC_R
            , private$raw_data@p
            , private$raw_data@i
            , private$raw_data@x
            , length(private$raw_data@p)
            , length(private$raw_data@x)
            , nrow(private$raw_data)
            , params_str
            , ref_handle
          )

        } else {

          # Unknown data type
          stop(
            "lgb.Dataset.construct: does not support constructing from "
            , sQuote(class(private$raw_data), q = FALSE)
          )

        }

      } else {

        # Reference is empty
        if (is.null(private$reference)) {
          stop("lgb.Dataset.construct: reference cannot be NULL for constructing data subset")
        }

        # Construct subset
        handle <- .Call(
          LGBM_DatasetGetSubset_R
          , ref_handle
          , c(private$used_indices)
          , length(private$used_indices)
          , params_str
        )

      }
      if (.is_null_handle(x = handle)) {
        stop("lgb.Dataset.construct: cannot create Dataset handle")
      }
      # Setup class and private type
      class(handle) <- "lgb.Dataset.handle"
      private$handle <- handle

      # Set feature names
      if (!is.null(private$colnames)) {
        self$set_colnames(colnames = private$colnames)
      }

      # Ensure that private$colnames matches the feature names on the C++ side. This line is necessary
      # in cases like constructing from a file or from a matrix with no column names.
      private$colnames <- .Call(
          LGBM_DatasetGetFeatureNames_R
          , private$handle
      )

      # Load init score if requested
      if (!is.null(private$predictor) && is.null(private$used_indices)) {

        # Setup initial scores
        init_score <- private$predictor$predict(
          data = private$raw_data
          , rawscore = TRUE
        )

        # Not needed to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score

      }

      # Should we free raw data?
      if (isTRUE(private$free_raw_data)) {
        private$raw_data <- NULL
      }

      # Get private information
      if (length(private$info) > 0L) {

        # Set infos
        for (i in seq_along(private$info)) {

          p <- private$info[i]
          self$set_field(
            field_name = names(p)
            , data = p[[1L]]
          )

        }

      }

      # Get label information existence
      if (is.null(self$get_field(field_name = "label"))) {
        stop("lgb.Dataset.construct: label should be set")
      }

      return(invisible(self))

    },

    # Dimension function
    dim = function() {

      # Check for handle
      if (!.is_null_handle(x = private$handle)) {

        num_row <- 0L
        num_col <- 0L

        # Get numeric data and numeric features
        .Call(
          LGBM_DatasetGetNumData_R
          , private$handle
          , num_row
        )
        .Call(
          LGBM_DatasetGetNumFeature_R
          , private$handle
          , num_col
        )
        return(
          c(num_row, num_col)
        )

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        # NOTE: requires Matrix package
        return(dim(private$raw_data))

      } else {

        # Trying to work with unknown dimensions is not possible
        stop(
          "dim: cannot get dimensions before dataset has been constructed, "
          , "please call lgb.Dataset.construct explicitly"
        )

      }

    },

    # Get number of bins for feature
    get_feature_num_bin = function(feature) {
      if (.is_null_handle(x = private$handle)) {
        stop("Cannot get number of bins in feature before constructing Dataset.")
      }
      if (is.character(feature)) {
        feature_name <- feature
        feature <- which(private$colnames == feature_name)
        if (length(feature) == 0L) {
          stop(sprintf("feature '%s' not found", feature_name))
        }
      }
      num_bin <- integer(1L)
      .Call(
        LGBM_DatasetGetFeatureNumBin_R
        , private$handle
        , feature - 1L
        , num_bin
      )
      return(num_bin)
    },

    # Get column names
    get_colnames = function() {

      # Check for handle
      if (!.is_null_handle(x = private$handle)) {
        private$colnames <- .Call(
          LGBM_DatasetGetFeatureNames_R
          , private$handle
        )
        return(private$colnames)

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        return(colnames(private$raw_data))

      } else {

        # Trying to work with unknown formats is not possible
        stop(
          "Dataset$get_colnames(): cannot get column names before dataset has been constructed, please call "
          , "lgb.Dataset.construct() explicitly"
        )

      }

    },

    # Set column names
    set_colnames = function(colnames) {

      # Check column names non-existence
      if (is.null(colnames)) {
        return(invisible(self))
      }

      # Check empty column names
      colnames <- as.character(colnames)
      if (length(colnames) == 0L) {
        return(invisible(self))
      }

      # Write column names
      private$colnames <- colnames
      if (!.is_null_handle(x = private$handle)) {

        # Merge names with tab separation
        merged_name <- paste(as.list(private$colnames), collapse = "\t")
        .Call(
          LGBM_DatasetSetFeatureNames_R
          , private$handle
          , merged_name
        )

      }

      return(invisible(self))

    },

    get_field = function(field_name) {

      # Check if attribute key is in the known attribute list
      if (!is.character(field_name) || length(field_name) != 1L || !field_name %in% .INFO_KEYS()) {
        stop(
          "Dataset$get_field(): field_name must be one of the following: "
          , toString(sQuote(.INFO_KEYS(), q = FALSE))
        )
      }

      # Check for info name and handle
      if (is.null(private$info[[field_name]])) {

        if (.is_null_handle(x = private$handle)) {
          stop("Cannot perform Dataset$get_field() before constructing Dataset.")
        }

        # Get field size of info
        info_len <- 0L
        .Call(
          LGBM_DatasetGetFieldSize_R
          , private$handle
          , field_name
          , info_len
        )

        if (info_len > 0L) {

          # Get back fields
          if (field_name == "group") {
            ret <- integer(info_len)
          } else {
            ret <- numeric(info_len)
          }

          .Call(
            LGBM_DatasetGetField_R
            , private$handle
            , field_name
            , ret
          )

          private$info[[field_name]] <- ret

        }
      }

      return(private$info[[field_name]])

    },

    set_field = function(field_name, data) {

      # Check if attribute key is in the known attribute list
      if (!is.character(field_name) || length(field_name) != 1L || !field_name %in% .INFO_KEYS()) {
        stop(
          "Dataset$set_field(): field_name must be one of the following: "
          , toString(sQuote(.INFO_KEYS(), q = FALSE))
        )
      }

      # Check for type of information
      data <- if (field_name == "group") {
        as.integer(data)
      } else {
        as.numeric(data)
      }

      # Store information privately
      private$info[[field_name]] <- data

      if (!.is_null_handle(x = private$handle) && !is.null(data)) {

        if (length(data) > 0L) {

          .Call(
            LGBM_DatasetSetField_R
            , private$handle
            , field_name
            , data
            , length(data)
          )

          private$version <- private$version + 1L

        }

      }

      return(invisible(self))

    },

    slice = function(idxset) {

      return(
        Dataset$new(
          data = NULL
          , params = private$params
          , reference = self
          , colnames = private$colnames
          , categorical_feature = private$categorical_feature
          , predictor = private$predictor
          , free_raw_data = private$free_raw_data
          , used_indices = sort(idxset, decreasing = FALSE)
        )
      )

    },

    # [description] Update Dataset parameters. If it has not been constructed yet,
    #               this operation just happens on the R side (updating private$params).
    #               If it has been constructed, parameters will be updated on the C++ side.
    update_params = function(params) {
      if (length(params) == 0L) {
        return(invisible(self))
      }
      new_params <- utils::modifyList(private$params, params)
      if (.is_null_handle(x = private$handle)) {
        private$params <- new_params
      } else {
        tryCatch({
          .Call(
            LGBM_DatasetUpdateParamChecking_R
            , .params2str(params = private$params)
            , .params2str(params = new_params)
          )
          private$params <- new_params
        }, error = function(e) {
          # If updating failed but raw data is not available, raise an error because
          # achieving what the user asked for is not possible
          if (is.null(private$raw_data)) {
            stop(e)
          }

          # If updating failed but raw data is available, modify the params
          # on the R side and re-set ("deconstruct") the Dataset
          private$params <- new_params
          private$finalize()
        })
      }
      return(invisible(self))

    },

    # [description] Get only Dataset-specific parameters. This is primarily used by
    #               Booster to update its parameters based on the characteristics of
    #               a Dataset. It should not be used by other methods in this class,
    #               since "verbose" is not a Dataset parameter and needs to be passed
    #               through to avoid globally re-setting verbosity.
    get_params = function() {
      dataset_params <- unname(unlist(.DATASET_PARAMETERS()))
      ret <- list()
      for (param_key in names(private$params)) {
        if (param_key %in% dataset_params) {
          ret[[param_key]] <- private$params[[param_key]]
        }
      }
      return(ret)
    },

    # Set categorical feature parameter
    set_categorical_feature = function(categorical_feature) {

      # Check for identical input
      if (identical(private$categorical_feature, categorical_feature)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_categorical_feature: cannot set categorical feature after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")
      }

      # Overwrite categorical features
      private$categorical_feature <- categorical_feature

      # Finalize and return self
      private$finalize()
      return(invisible(self))

    },

    set_reference = function(reference) {

      # setting reference to this same Dataset object doesn't require any changes
      if (identical(private$reference, reference)) {
        return(invisible(self))
      }

      # changing the reference removes the Dataset object on the C++ side, so it should only
      # be done if you still have the raw_data available, so that the new Dataset can be reconstructed
      if (is.null(private$raw_data)) {
        stop("set_reference: cannot set reference after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")
      }

      if (!.is_Dataset(reference)) {
        stop("set_reference: Can only use lgb.Dataset as a reference")
      }

      # Set known references
      self$set_categorical_feature(categorical_feature = reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(colnames = reference$get_colnames())
      private$set_predictor(predictor = reference$.__enclos_env__$private$predictor)

      # Store reference
      private$reference <- reference

      # Finalize and return self
      private$finalize()
      return(invisible(self))

    },

    # Save binary model
    save_binary = function(fname) {

      # Store binary data
      self$construct()
      .Call(
        LGBM_DatasetSaveBinary_R
        , private$handle
        , path.expand(fname)
      )
      return(invisible(self))
    }

  ),
  private = list(
    handle = NULL,
    raw_data = NULL,
    params = list(),
    reference = NULL,
    colnames = NULL,
    categorical_feature = NULL,
    predictor = NULL,
    free_raw_data = TRUE,
    used_indices = NULL,
    info = NULL,
    version = 0L,

    # finalize() will free up the handles
    finalize = function() {
      .Call(
        LGBM_DatasetFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },

    get_handle = function() {

      # Get handle and construct if needed
      if (.is_null_handle(x = private$handle)) {
        self$construct()
      }
      return(private$handle)

    },

    set_predictor = function(predictor) {

      if (identical(private$predictor, predictor)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_predictor: cannot set predictor after free raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")
      }

      # Check for empty predictor
      if (!is.null(predictor)) {

        # Predictor is unknown
        if (!.is_Predictor(predictor)) {
          stop("set_predictor: Can only use lgb.Predictor as predictor")
        }

      }

      # Store predictor
      private$predictor <- predictor

      # Finalize and return self
      private$finalize()
      return(invisible(self))

    }

  )
)

#' @title Construct \code{lgb.Dataset} object
#' @description LightGBM does not train on raw data.
#'              It discretizes continuous features into histogram bins, tries to
#'              combine categorical features, and automatically handles missing and
#               infinite values.
#'
#'              The \code{Dataset} class handles that preprocessing, and holds that
#'              alternative representation of the input data.
#' @inheritParams lgb_shared_dataset_params
#' @param data a \code{matrix} object, a \code{dgCMatrix} object,
#'             a character representing a path to a text file (CSV, TSV, or LibSVM),
#'             or a character representing a path to a binary \code{lgb.Dataset} file
#' @param params a list of parameters. See
#'               \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#dataset-parameters}{
#'               The "Dataset Parameters" section of the documentation} for a list of parameters
#'               and valid values.
#' @param reference reference dataset. When LightGBM creates a Dataset, it does some preprocessing like binning
#'                  continuous features into histograms. If you want to apply the same bin boundaries from an existing
#'                  dataset to new \code{data}, pass that existing Dataset to this argument.
#' @param colnames names of columns
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @param free_raw_data LightGBM constructs its data format, called a "Dataset", from tabular data.
#'                      By default, that Dataset object on the R side does not keep a copy of the raw data.
#'                      This reduces LightGBM's memory consumption, but it means that the Dataset object
#'                      cannot be changed after it has been constructed. If you'd prefer to be able to
#'                      change the Dataset object after construction, set \code{free_raw_data = FALSE}.
#'
#' @return constructed dataset
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' lgb.Dataset.save(dtrain, data_file)
#' dtrain <- lgb.Dataset(data_file)
#' lgb.Dataset.construct(dtrain)
#' }
#' @export
lgb.Dataset <- function(data,
                        params = list(),
                        reference = NULL,
                        colnames = NULL,
                        categorical_feature = NULL,
                        free_raw_data = TRUE,
                        label = NULL,
                        weight = NULL,
                        group = NULL,
                        init_score = NULL) {

  return(
    invisible(Dataset$new(
      data = data
      , params = params
      , reference = reference
      , colnames = colnames
      , categorical_feature = categorical_feature
      , predictor = NULL
      , free_raw_data = free_raw_data
      , used_indices = NULL
      , label = label
      , weight = weight
      , group = group
      , init_score = init_score
    ))
  )

}

#' @name lgb.Dataset.create.valid
#' @title Construct validation data
#' @description Construct validation data according to training data
#' @inheritParams lgb_shared_dataset_params
#' @param dataset \code{lgb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object,
#'             a character representing a path to a text file (CSV, TSV, or LibSVM),
#'             or a character representing a path to a binary \code{Dataset} file
#' @param params a list of parameters. See
#'               \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#dataset-parameters}{
#'               The "Dataset Parameters" section of the documentation} for a list of parameters
#'               and valid values. If this is an empty list (the default), the validation Dataset
#'               will have the same parameters as the Dataset passed to argument \code{dataset}.
#'
#' @return constructed dataset
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
#'
#' # parameters can be changed between the training data and validation set,
#' # for example to account for training data in a text file with a header row
#' # and validation data in a text file without it
#' train_file <- tempfile(pattern = "train_", fileext = ".csv")
#' write.table(
#'   data.frame(y = rnorm(100L), x1 = rnorm(100L), x2 = rnorm(100L))
#'   , file = train_file
#'   , sep = ","
#'   , col.names = TRUE
#'   , row.names = FALSE
#'   , quote = FALSE
#' )
#'
#' valid_file <- tempfile(pattern = "valid_", fileext = ".csv")
#' write.table(
#'   data.frame(y = rnorm(100L), x1 = rnorm(100L), x2 = rnorm(100L))
#'   , file = valid_file
#'   , sep = ","
#'   , col.names = FALSE
#'   , row.names = FALSE
#'   , quote = FALSE
#' )
#'
#' dtrain <- lgb.Dataset(
#'   data = train_file
#'   , params = list(has_header = TRUE)
#' )
#' dtrain$construct()
#'
#' dvalid <- lgb.Dataset(
#'   data = valid_file
#'   , params = list(has_header = FALSE)
#' )
#' dvalid$construct()
#' }
#' @export
lgb.Dataset.create.valid <- function(dataset,
                                     data,
                                     label = NULL,
                                     weight = NULL,
                                     group = NULL,
                                     init_score = NULL,
                                     params = list()) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.Dataset.create.valid: input data should be an lgb.Dataset object")
  }

  # Create validation dataset
  return(invisible(
    dataset$create_valid(
      data = data
      , label = label
      , weight = weight
      , group = group
      , init_score = init_score
      , params = params
    )
  ))

}

#' @name lgb.Dataset.construct
#' @title Construct Dataset explicitly
#' @description Construct Dataset explicitly
#' @param dataset Object of class \code{lgb.Dataset}
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#' }
#' @return constructed dataset
#' @export
lgb.Dataset.construct <- function(dataset) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.Dataset.construct: input data should be an lgb.Dataset object")
  }

  return(invisible(dataset$construct()))

}

#' @title Dimensions of an \code{lgb.Dataset}
#' @description Returns a vector of numbers of rows and of columns in an \code{lgb.Dataset}.
#' @param x Object of class \code{lgb.Dataset}
#'
#' @return a vector of numbers of rows and of columns
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{lgb.Dataset} object.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#' }
#' @rdname dim
#' @export
dim.lgb.Dataset <- function(x) {

  if (!.is_Dataset(x = x)) {
    stop("dim.lgb.Dataset: input data should be an lgb.Dataset object")
  }

  return(x$dim())

}

#' @title Handling of column names of \code{lgb.Dataset}
#' @description Only column names are supported for \code{lgb.Dataset}, thus setting of
#'              row names would have no effect and returned row names would be NULL.
#' @param x object of class \code{lgb.Dataset}
#' @param value a list of two elements: the first one is ignored
#'              and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(seq_len(ncol(train$data)))
#' print(dtrain, verbose = TRUE)
#' }
#' @rdname dimnames.lgb.Dataset
#' @return A list with the dimension names of the dataset
#' @export
dimnames.lgb.Dataset <- function(x) {

  if (!.is_Dataset(x = x)) {
    stop("dimnames.lgb.Dataset: input data should be an lgb.Dataset object")
  }

  # Return dimension names
  return(list(NULL, x$get_colnames()))

}

#' @rdname dimnames.lgb.Dataset
#' @export
`dimnames<-.lgb.Dataset` <- function(x, value) {

  # Check if invalid element list
  if (!identical(class(value), "list") || length(value) != 2L) {
    stop("invalid ", sQuote("value", q = FALSE), " given: must be a list of two elements")
  }

  # Check for unknown row names
  if (!is.null(value[[1L]])) {
    stop("lgb.Dataset does not have rownames")
  }

  if (is.null(value[[2L]])) {

    x$set_colnames(colnames = NULL)
    return(x)

  }

  # Check for unmatching column size
  if (ncol(x) != length(value[[2L]])) {
    stop(
      "can't assign "
      , sQuote(length(value[[2L]]), q = FALSE)
      , " colnames to an lgb.Dataset with "
      , sQuote(ncol(x), q = FALSE)
      , " columns"
    )
  }

  # Set column names properly, and return
  x$set_colnames(colnames = value[[2L]])
  return(x)

}

#' @title Slice a dataset
#' @description Get a new \code{lgb.Dataset} containing the specified rows of
#'              original \code{lgb.Dataset} object
#'
#'              \emph{Renamed from} \code{slice()} \emph{in 4.4.0}
#'
#' @param dataset Object of class \code{lgb.Dataset}
#' @param idxset an integer vector of indices of rows needed
#' @return constructed sub dataset
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' dsub <- lgb.slice.Dataset(dtrain, seq_len(42L))
#' lgb.Dataset.construct(dsub)
#' labels <- lightgbm::get_field(dsub, "label")
#' }
#' @export
lgb.slice.Dataset <- function(dataset, idxset) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.slice.Dataset: input dataset should be an lgb.Dataset object")
  }

  return(invisible(dataset$slice(idxset = idxset)))

}

#' @name get_field
#' @title Get one attribute of a \code{lgb.Dataset}
#' @description Get one attribute of a \code{lgb.Dataset}
#' @param dataset Object of class \code{lgb.Dataset}
#' @param field_name String with the name of the attribute to get. One of the following.
#' \itemize{
#'     \item \code{label}: label lightgbm learns from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from.
#' }
#' @return requested attribute
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#'
#' labels <- lightgbm::get_field(dtrain, "label")
#' lightgbm::set_field(dtrain, "label", 1 - labels)
#'
#' labels2 <- lightgbm::get_field(dtrain, "label")
#' stopifnot(all(labels2 == 1 - labels))
#' }
#' @export
get_field <- function(dataset, field_name) {
  UseMethod("get_field")
}

#' @rdname get_field
#' @export
get_field.lgb.Dataset <- function(dataset, field_name) {

  # Check if dataset is not a dataset
  if (!.is_Dataset(x = dataset)) {
    stop("get_field.lgb.Dataset(): input dataset should be an lgb.Dataset object")
  }

  return(dataset$get_field(field_name = field_name))

}

#' @name set_field
#' @title Set one attribute of a \code{lgb.Dataset} object
#' @description Set one attribute of a \code{lgb.Dataset}
#' @param dataset Object of class \code{lgb.Dataset}
#' @param field_name String with the name of the attribute to set. One of the following.
#' \itemize{
#'     \item \code{label}: label lightgbm learns from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from.
#' }
#' @param data The data for the field. See examples.
#' @return The \code{lgb.Dataset} you passed in.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#'
#' labels <- lightgbm::get_field(dtrain, "label")
#' lightgbm::set_field(dtrain, "label", 1 - labels)
#'
#' labels2 <- lightgbm::get_field(dtrain, "label")
#' stopifnot(all.equal(labels2, 1 - labels))
#' }
#' @export
set_field <- function(dataset, field_name, data) {
  UseMethod("set_field")
}

#' @rdname set_field
#' @export
set_field.lgb.Dataset <- function(dataset, field_name, data) {

  if (!.is_Dataset(x = dataset)) {
    stop("set_field.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }

  return(invisible(dataset$set_field(field_name = field_name, data = data)))
}

#' @name lgb.Dataset.set.categorical
#' @title Set categorical feature of \code{lgb.Dataset}
#' @description Set the categorical features of an \code{lgb.Dataset} object. Use this function
#'              to tell LightGBM which features should be treated as categorical.
#' @param dataset object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' lgb.Dataset.save(dtrain, data_file)
#' dtrain <- lgb.Dataset(data_file)
#' lgb.Dataset.set.categorical(dtrain, 1L:2L)
#' }
#' @rdname lgb.Dataset.set.categorical
#' @export
lgb.Dataset.set.categorical <- function(dataset, categorical_feature) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.Dataset.set.categorical: input dataset should be an lgb.Dataset object")
  }

  return(invisible(dataset$set_categorical_feature(categorical_feature = categorical_feature)))

}

#' @name lgb.Dataset.set.reference
#' @title Set reference of \code{lgb.Dataset}
#' @description If you want to use validation data, you should set reference to training data
#' @param dataset object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' # create training Dataset
#' data(agaricus.train, package ="lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' # create a validation Dataset, using dtrain as a reference
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset(test$data, label = test$label)
#' lgb.Dataset.set.reference(dtest, dtrain)
#' }
#' @rdname lgb.Dataset.set.reference
#' @export
lgb.Dataset.set.reference <- function(dataset, reference) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.Dataset.set.reference: input dataset should be an lgb.Dataset object")
  }

  return(invisible(dataset$set_reference(reference = reference)))
}

#' @name lgb.Dataset.save
#' @title Save \code{lgb.Dataset} to a binary file
#' @description Please note that \code{init_score} is not saved in binary file.
#'              If you need it, please set it again after loading Dataset.
#' @param dataset object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.save(dtrain, tempfile(fileext = ".bin"))
#' }
#' @export
lgb.Dataset.save <- function(dataset, fname) {

  if (!.is_Dataset(x = dataset)) {
    stop("lgb.Dataset.save: input dataset should be an lgb.Dataset object")
  }

  if (!is.character(fname)) {
    stop("lgb.Dataset.save: fname should be a character or a file connection")
  }

  return(invisible(dataset$save_binary(fname = fname)))
}
