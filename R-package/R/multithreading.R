#' @name setLGBMThreads
#' @title Set maximum number of threads used by LightGBM
#' @description LightGBM attempts to speed up many operations by using multi-threading.
#'              The number of threads used in those operations can be controlled via the
#'              \code{num_threads} parameter passed through \code{params} to functions like
#'              \link{lgb.train} and \link{lgb.Dataset}. However, some operations (like materializing
#'              a model from a text file) are done via code paths that don't explicitly accept thread-control
#'              configuration.
#'
#'              Use this function to set the maximum number of threads LightGBM will use for such operations.
#'
#'              This function affects all LightGBM operations in the same process.
#'
#'              So, for example, if you call \code{setLGBMthreads(4)}, no other multi-threaded LightGBM
#'              operation in the same process will use more than 4 threads.
#'
#'              Call \code{setLGBMthreads(-1)} to remove this limitation.
#' @param num_threads maximum number of threads to be used by LightGBM in multi-threaded operations
#' @return NULL
#' @seealso \link{getLGBMthreads}
#' @export
setLGBMthreads <- function(num_threads) {
    .Call(
        LGBM_SetMaxThreads_R,
        num_threads
    )
    return(invisible(NULL))
}

#' @name getLGBMThreads
#' @title Get default number of threads used by LightGBM
#' @description LightGBM attempts to speed up many operations by using multi-threading.
#'              The number of threads used in those operations can be controlled via the
#'              \code{num_threads} parameter passed through \code{params} to functions like
#'              \link{lgb.train} and \link{lgb.Dataset}. However, some operations (like materializing
#'              a model from a text file) are done via code paths that don't explicitly accept thread-control
#'              configuration.
#'
#'              Use this function to see the default number of threads LightGBM will use for such operations.
#' @return number of threads as an integer. \code{-1} means that in situations where parameter \code{num_threads} is
#'         not explicitly supplied, LightGBM will choose a number of threads to use automatically.
#' @seealso \link{setLGBMthreads}
#' @export
getLGBMthreads <- function() {
    out <- 0L
    .Call(
        LGBM_GetMaxThreads_R,
        out
    )
    return(out)
}
