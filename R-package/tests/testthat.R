library(testthat)
library(lightgbm)  # nolint: unused_import.

test_check(
    package = "lightgbm"
    , stop_on_failure = TRUE
    , stop_on_warning = FALSE
    , reporter = testthat::SummaryReporter$new()
)
