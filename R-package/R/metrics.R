# [description] List of metrics known to LightGBM. The most up to date list can be found
#               at https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
#
# [return] A named logical vector, where each key is a metric name and each value is a boolean.
#          TRUE if higher values of the metric are desirable, FALSE if lower values are desirable.
#          Note that only the 'main' metrics are stored here, not aliases, since only the 'main' metrics
#          are returned from the C++ side. For example, if you use `metric = "mse"` in your code,
#          the metric name `"l2"` will be returned.
.METRICS_HIGHER_BETTER <- function() {
    return(
        c(
            "l1" = FALSE
            , "l2" = FALSE
            , "mape" = FALSE
            , "rmse" = FALSE
            , "quantile" = FALSE
            , "huber" = FALSE
            , "fair" = FALSE
            , "poisson" = FALSE
            , "gamma" = FALSE
            , "gamma_deviance" = FALSE
            , "tweedie" = FALSE
            , "ndcg" = TRUE
            , "map" = TRUE
            , "auc" = TRUE
            , "average_precision" = TRUE
            , "r2" = TRUE
            , "binary_logloss" = FALSE
            , "binary_error" = FALSE
            , "auc_mu" = TRUE
            , "multi_logloss" = FALSE
            , "multi_error" = FALSE
            , "cross_entropy" = FALSE
            , "cross_entropy_lambda" = FALSE
            , "kullback_leibler" = FALSE
        )
    )
}
