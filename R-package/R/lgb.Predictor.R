#' @importFrom methods is new
#' @importFrom R6 R6Class
#' @importFrom utils read.delim
#' @importClassesFrom Matrix dsparseMatrix dsparseVector dgCMatrix dgRMatrix CsparseMatrix RsparseMatrix
Predictor <- R6::R6Class(

  classname = "lgb.Predictor",
  cloneable = FALSE,
  public = list(

    # Initialize will create a starter model
    initialize = function(modelfile, params = list(), fast_predict_config = list()) {
      private$params <- .params2str(params = params)
      handle <- NULL

      if (is.character(modelfile)) {

        # Create handle on it
        handle <- .Call(
          LGBM_BoosterCreateFromModelfile_R
          , path.expand(modelfile)
        )
        private$need_free_handle <- TRUE

      } else if (methods::is(modelfile, "lgb.Booster.handle") || inherits(modelfile, "externalptr")) {

        # Check if model file is a booster handle already
        handle <- modelfile
        private$need_free_handle <- FALSE

      } else if (.is_Booster(modelfile)) {

        handle <- modelfile$get_handle()
        private$need_free_handle <- FALSE

      } else {

        stop("lgb.Predictor: modelfile must be either a character filename or an lgb.Booster.handle")

      }

      private$fast_predict_config <- fast_predict_config

      # Override class and store it
      class(handle) <- "lgb.Booster.handle"
      private$handle <- handle

      return(invisible(NULL))

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      .Call(
        LGBM_BoosterGetCurrentIteration_R
        , private$handle
        , cur_iter
      )
      return(cur_iter)

    },

    # Predict from data
    predict = function(data,
                       start_iteration = NULL,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE) {

      # Check if number of iterations is existing - if not, then set it to -1 (use all)
      if (is.null(num_iteration)) {
        num_iteration <- -1L
      }
      # Check if start iterations is existing - if not, then set it to 0 (start from the first iteration)
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }

      # Check if data is a file name and not a matrix
      if (identical(class(data), "character") && length(data) == 1L) {

        data <- path.expand(data)

        # Data is a filename, create a temporary file with a "lightgbm_" pattern in it
        tmp_filename <- tempfile(pattern = "lightgbm_")
        on.exit(unlink(tmp_filename), add = TRUE)

        # Predict from temporary file
        .Call(
          LGBM_BoosterPredictForFile_R
          , private$handle
          , data
          , as.integer(header)
          , as.integer(rawscore)
          , as.integer(predleaf)
          , as.integer(predcontrib)
          , as.integer(start_iteration)
          , as.integer(num_iteration)
          , private$params
          , tmp_filename
        )

        # Get predictions from file
        preds <- utils::read.delim(tmp_filename, header = FALSE, sep = "\t")
        num_row <- nrow(preds)
        preds <- as.vector(t(preds))

      } else if (predcontrib && inherits(data, c("dsparseMatrix", "dsparseVector"))) {

        ncols <- .Call(LGBM_BoosterGetNumFeature_R, private$handle)
        ncols_out <- integer(1L)
        .Call(LGBM_BoosterGetNumClasses_R, private$handle, ncols_out)
        ncols_out <- (ncols + 1L) * max(ncols_out, 1L)
        if (is.na(ncols_out)) {
          ncols_out <- as.numeric(ncols + 1L) * as.numeric(max(ncols_out, 1L))
        }
        if (!inherits(data, "dsparseVector") && ncols_out > .Machine$integer.max) {
          stop("Resulting matrix of feature contributions is too large for R to handle.")
        }

        if (inherits(data, "dsparseVector")) {

          if (length(data) > ncols) {
            stop(sprintf("Model was fitted to data with %d columns, input data has %.0f columns."
                         , ncols
                         , length(data)))
          }
          res <- .Call(
            LGBM_BoosterPredictSparseOutput_R
            , private$handle
            , c(0L, as.integer(length(data@x)))
            , data@i - 1L
            , data@x
            , TRUE
            , 1L
            , ncols
            , start_iteration
            , num_iteration
            , private$params
          )
          out <- methods::new("dsparseVector")
          out@i <- res$indices + 1L
          out@x <- res$data
          out@length <- ncols_out
          return(out)

        } else if (inherits(data, "dgRMatrix")) {

          if (ncol(data) > ncols) {
            stop(sprintf("Model was fitted to data with %d columns, input data has %.0f columns."
                         , ncols
                         , ncol(data)))
          }
          res <- .Call(
            LGBM_BoosterPredictSparseOutput_R
            , private$handle
            , data@p
            , data@j
            , data@x
            , TRUE
            , nrow(data)
            , ncols
            , start_iteration
            , num_iteration
            , private$params
          )
          out <- methods::new("dgRMatrix")
          out@p <- res$indptr
          out@j <- res$indices
          out@x <- res$data
          out@Dim <- as.integer(c(nrow(data), ncols_out))

        } else if (inherits(data, "dgCMatrix")) {

          if (ncol(data) != ncols) {
            stop(sprintf("Model was fitted to data with %d columns, input data has %.0f columns."
                         , ncols
                         , ncol(data)))
          }
          res <- .Call(
            LGBM_BoosterPredictSparseOutput_R
            , private$handle
            , data@p
            , data@i
            , data@x
            , FALSE
            , nrow(data)
            , ncols
            , start_iteration
            , num_iteration
            , private$params
          )
          out <- methods::new("dgCMatrix")
          out@p <- res$indptr
          out@i <- res$indices
          out@x <- res$data
          out@Dim <- as.integer(c(nrow(data), length(res$indptr) - 1L))

        } else {

          stop(sprintf("Predictions on sparse inputs are only allowed for '%s', '%s', '%s' - got: %s"
                       , "dsparseVector"
                       , "dgRMatrix"
                       , "dgCMatrix"
                       , toString(class(data))))
        }

        if (NROW(row.names(data))) {
          out@Dimnames[[1L]] <- row.names(data)
        }
        return(out)

      } else {

        # Not a file, we need to predict from R object
        num_row <- nrow(data)
        if (is.null(num_row)) {
          num_row <- 1L
        }

        npred <- 0L

        # Check number of predictions to do
        .Call(
          LGBM_BoosterCalcNumPredict_R
          , private$handle
          , as.integer(num_row)
          , as.integer(rawscore)
          , as.integer(predleaf)
          , as.integer(predcontrib)
          , as.integer(start_iteration)
          , as.integer(num_iteration)
          , npred
        )

        # Pre-allocate empty vector
        preds <- numeric(npred)

        # Check if data is a matrix
        if (is.matrix(data)) {
          # this if() prevents the memory and computational costs
          # of converting something that is already "double" to "double"
          if (storage.mode(data) != "double") {
            storage.mode(data) <- "double"
          }

          if (nrow(data) == 1L) {

            use_fast_config <- private$check_can_use_fast_predict_config(
              csr = FALSE
              , rawscore = rawscore
              , predleaf = predleaf
              , predcontrib = predcontrib
              , start_iteration = start_iteration
              , num_iteration = num_iteration
            )

            if (use_fast_config) {
              .Call(
                LGBM_BoosterPredictForMatSingleRowFast_R
                , private$fast_predict_config$handle
                , data
                , preds
              )
            } else {
              .Call(
                LGBM_BoosterPredictForMatSingleRow_R
                , private$handle
                , data
                , rawscore
                , predleaf
                , predcontrib
                , start_iteration
                , num_iteration
                , private$params
                , preds
              )
            }

          } else {
            .Call(
              LGBM_BoosterPredictForMat_R
              , private$handle
              , data
              , as.integer(nrow(data))
              , as.integer(ncol(data))
              , as.integer(rawscore)
              , as.integer(predleaf)
              , as.integer(predcontrib)
              , as.integer(start_iteration)
              , as.integer(num_iteration)
              , private$params
              , preds
            )
          }

        } else if (inherits(data, "dsparseVector")) {

          if (length(self$fast_predict_config)) {
            ncols <- self$fast_predict_config$ncols
            use_fast_config <- private$check_can_use_fast_predict_config(
                csr = TRUE
                , rawscore = rawscore
                , predleaf = predleaf
                , predcontrib = predcontrib
                , start_iteration = start_iteration
                , num_iteration = num_iteration
              )
          } else {
            ncols <- .Call(LGBM_BoosterGetNumFeature_R, private$handle)
            use_fast_config <- FALSE
          }

          if (length(data) > ncols) {
            stop(sprintf("Model was fitted to data with %d columns, input data has %.0f columns."
                         , ncols
                         , length(data)))
          }

          if (use_fast_config) {
            .Call(
              LGBM_BoosterPredictForCSRSingleRowFast_R
              , self$fast_predict_config$handle
              , data@i - 1L
              , data@x
              , preds
            )
          } else {
            .Call(
              LGBM_BoosterPredictForCSRSingleRow_R
              , private$handle
              , data@i - 1L
              , data@x
              , ncols
              , as.integer(rawscore)
              , as.integer(predleaf)
              , as.integer(predcontrib)
              , start_iteration
              , num_iteration
              , private$params
              , preds
            )
          }

        } else if (inherits(data, "dgRMatrix")) {

          ncols <- .Call(LGBM_BoosterGetNumFeature_R, private$handle)
          if (ncol(data) > ncols) {
            stop(sprintf("Model was fitted to data with %d columns, input data has %.0f columns."
                         , ncols
                         , ncol(data)))
          }

          if (nrow(data) == 1L) {

            if (length(self$fast_predict_config)) {
              ncols <- self$fast_predict_config$ncols
              use_fast_config <- private$check_can_use_fast_predict_config(
                csr = TRUE
                , rawscore = rawscore
                , predleaf = predleaf
                , predcontrib = predcontrib
                , start_iteration = start_iteration
                , num_iteration = num_iteration
              )
            } else {
              ncols <- .Call(LGBM_BoosterGetNumFeature_R, private$handle)
              use_fast_config <- FALSE
            }

            if (use_fast_config) {
              .Call(
                LGBM_BoosterPredictForCSRSingleRowFast_R
                , self$fast_predict_config$handle
                , data@j
                , data@x
                , preds
              )
            } else {
              .Call(
                LGBM_BoosterPredictForCSRSingleRow_R
                , private$handle
                , data@j
                , data@x
                , ncols
                , as.integer(rawscore)
                , as.integer(predleaf)
                , as.integer(predcontrib)
                , start_iteration
                , num_iteration
                , private$params
                , preds
              )
            }

          } else {

            .Call(
              LGBM_BoosterPredictForCSR_R
              , private$handle
              , data@p
              , data@j
              , data@x
              , ncols
              , as.integer(rawscore)
              , as.integer(predleaf)
              , as.integer(predcontrib)
              , start_iteration
              , num_iteration
              , private$params
              , preds
            )

          }

        } else if (methods::is(data, "dgCMatrix")) {
          if (length(data@p) > 2147483647L) {
            stop("Cannot support large CSC matrix")
          }
          # Check if data is a dgCMatrix (sparse matrix, column compressed format)
          .Call(
            LGBM_BoosterPredictForCSC_R
            , private$handle
            , data@p
            , data@i
            , data@x
            , length(data@p)
            , length(data@x)
            , nrow(data)
            , as.integer(rawscore)
            , as.integer(predleaf)
            , as.integer(predcontrib)
            , as.integer(start_iteration)
            , as.integer(num_iteration)
            , private$params
            , preds
          )

        } else {

          stop("predict: cannot predict on data of class ", sQuote(class(data), q = FALSE))

        }
      }

      # Check if number of rows is strange (not a multiple of the dataset rows)
      if (length(preds) %% num_row != 0L) {
        stop(
          "predict: prediction length "
          , sQuote(length(preds), q = FALSE)
          , " is not a multiple of nrows(data): "
          , sQuote(num_row, q = FALSE)
        )
      }

      # Get number of cases per row
      npred_per_case <- length(preds) / num_row

      # Data reshaping
      if (npred_per_case > 1L || predleaf || predcontrib) {
        preds <- matrix(preds, ncol = npred_per_case, byrow = TRUE)
      }

      # Keep row names if possible
      if (NROW(row.names(data)) && NROW(data) == NROW(preds)) {
        if (is.null(dim(preds))) {
          names(preds) <- row.names(data)
        } else {
          row.names(preds) <- row.names(data)
        }
      }

      return(preds)
    }

  ),
  private = list(
    handle = NULL
    , need_free_handle = FALSE
    , params = ""
    , fast_predict_config = list()
    , check_can_use_fast_predict_config = function(csr,
                                                   rawscore,
                                                   predleaf,
                                                   predcontrib,
                                                   start_iteration,
                                                   num_iteration) {

      if (!NROW(private$fast_predict_config)) {
        return(FALSE)
      }

      if (.is_null_handle(private$fast_predict_config$handle)) {
        warning(paste0("Model had fast CSR predict configuration, but it is inactive."
                       , " Try re-generating it through 'lgb.configure_fast_predict'."))
        return(FALSE)
      }

      if (isTRUE(csr) != private$fast_predict_config$csr) {
        return(FALSE)
      }

      return(
        private$params == "" &&
        private$fast_predict_config$rawscore == rawscore &&
        private$fast_predict_config$predleaf == predleaf &&
        private$fast_predict_config$predcontrib == predcontrib &&
        .equal_or_both_null(private$fast_predict_config$start_iteration, start_iteration) &&
        .equal_or_both_null(private$fast_predict_config$num_iteration, num_iteration)
      )
    }

    # finalize() will free up the handles
    , finalize = function() {
      if (private$need_free_handle) {
        .Call(
          LGBM_BoosterFree_R
          , private$handle
        )
        private$handle <- NULL
      }
      return(invisible(NULL))
    }
  )
)
