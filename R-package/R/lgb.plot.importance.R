#' @name lgb.plot.importance
#' @title Plot feature importance as a bar graph
#' @description Plot previously calculated feature importance: Gain, Cover and Frequency, as a bar graph.
#' @param tree_imp a \code{data.table} returned by \code{\link{lgb.importance}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param measure the name of importance measure to plot, can be "Gain", "Cover" or "Frequency".
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{\link[graphics]{barplot}}.
#'            Set a number smaller than 1.0 to make the bar labels smaller than R's default and values
#'            greater than 1.0 to make them larger.
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the defined importance of a feature.
#' Features are shown ranked in a decreasing importance order.
#'
#' @return
#' The \code{lgb.plot.importance} function creates a \code{barplot}
#' and silently returns a processed data.table with \code{top_n} features sorted by defined importance.
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
#'     objective = "binary"
#'     , learning_rate = 0.1
#'     , min_data_in_leaf = 1L
#'     , min_sum_hessian_in_leaf = 1.0
#'     , num_threads = 2L
#' )
#'
#' model <- lgb.train(
#'     params = params
#'     , data = dtrain
#'     , nrounds = 5L
#' )
#'
#' tree_imp <- lgb.importance(model, percentage = TRUE)
#' lgb.plot.importance(tree_imp, top_n = 5L, measure = "Gain")
#' }
#' @importFrom graphics barplot par
#' @export
lgb.plot.importance <- function(tree_imp,
                                top_n = 10L,
                                measure = "Gain",
                                left_margin = 10L,
                                cex = NULL
                                ) {

  # Check for measurement (column names) correctness
  measure <- match.arg(
    measure
    , choices = c("Gain", "Cover", "Frequency")
    , several.ok = FALSE
  )

  # Get top N importance (defaults to 10)
  top_n <- min(top_n, nrow(tree_imp))

  # Parse importance
  tree_imp <- tree_imp[order(abs(get(measure)), decreasing = TRUE), ][seq_len(top_n), ]

  # Attempt to setup a correct cex
  if (is.null(cex)) {
    cex <- 2.5 / log2(1.0 + top_n)
  }

  # Refresh plot
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op))

  graphics::par(
    mar = c(
      op$mar[1L]
      , left_margin
      , op$mar[3L]
      , op$mar[4L]
    )
  )

  tree_imp[rev(seq_len(.N)),
           graphics::barplot(
               height = get(measure)
               , names.arg = Feature
               , horiz = TRUE
               , border = NA
               , main = "Feature Importance"
               , xlab = measure
               , cex.names = cex
               , las = 1L
           )]

  return(invisible(tree_imp))

}
