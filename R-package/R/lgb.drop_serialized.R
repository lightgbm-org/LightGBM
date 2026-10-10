#' @name lgb.drop_serialized
#' @title Drop serialized raw bytes in a LightGBM model object
#' @description If a LightGBM model object was produced with argument `serializable=TRUE`, the R object will keep
#' a copy of the underlying C++ object as raw bytes, which can be used to reconstruct such object after getting
#' serialized and de-serialized, but at the cost of extra memory usage. If these raw bytes are not needed anymore,
#' they can be dropped through this function in order to save memory. Note that the object will be modified in-place.
#'
#'              \emph{New in version 4.0.0}
#'
#' @param model \code{lgb.Booster} object which was produced with `serializable=TRUE`.
#'
#' @return \code{lgb.Booster} (the same `model` object that was passed as input, as invisible).
#' @seealso \link{lgb.restore_handle}, \link{lgb.make_serializable}.
#' @export
lgb.drop_serialized <- function(model) {
  if (!.is_Booster(x = model)) {
    stop("lgb.drop_serialized: model should be an ", sQuote("lgb.Booster", q = FALSE))
  }
  model$drop_raw()
  return(invisible(model))
}
