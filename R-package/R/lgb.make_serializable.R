#' @name lgb.make_serializable
#' @title Make a LightGBM object serializable by keeping raw bytes
#' @description If a LightGBM model object was produced with argument `serializable=FALSE`, the R object will not
#' be serializable (e.g. cannot save and load with \code{saveRDS} and \code{readRDS}) as it will lack the raw bytes
#' needed to reconstruct its underlying C++ object. This function can be used to forcibly produce those serialized
#' raw bytes and make the object serializable. Note that the object will be modified in-place.
#'
#'              \emph{New in version 4.0.0}
#'
#' @param model \code{lgb.Booster} object which was produced with `serializable=FALSE`.
#'
#' @return \code{lgb.Booster} (the same `model` object that was passed as input, as invisible).
#' @seealso \link{lgb.restore_handle}, \link{lgb.drop_serialized}.
#' @export
lgb.make_serializable <- function(model) {
  if (!.is_Booster(x = model)) {
    stop("lgb.make_serializable: model should be an ", sQuote("lgb.Booster", q = FALSE))
  }
  model$save_raw()
  return(invisible(model))
}
