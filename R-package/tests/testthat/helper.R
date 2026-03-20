# ref for this file:
#
# * https://r-pkgs.org/testing-design.html#testthat-helper-files
# * https://r-pkgs.org/testing-design.html#testthat-setup-files

# LightGBM-internal fix to comply with CRAN policy of only using up to 2 threads in tests and example.
#
# per https://cran.r-project.org/web/packages/policies.html
#
# > If running a package uses multiple threads/cores it must never use more than two simultaneously:
#   the check farm is a shared resource and will typically be running many checks simultaneously.
#
.LGB_MAX_THREADS <- 2L
setLGBMthreads(.LGB_MAX_THREADS)

# control data.table parallelism
# ref: https://github.com/Rdatatable/data.table/issues/5658
data.table::setDTthreads(1L)

# by default, how much should results in tests be allowed to differ from hard-coded expected numbers?
.LGB_NUMERIC_TOLERANCE <- 1e-6

# are the tests running on Windows?
.LGB_ON_WINDOWS <- .Platform$OS.type == "windows"
.LGB_ON_32_BIT_WINDOWS <- .LGB_ON_WINDOWS && .Machine$sizeof.pointer != 8L

# are the tests running in a UTF-8 locale?
.LGB_UTF8_LOCALE <- all(endsWith(
  Sys.getlocale(category = "LC_CTYPE")
  , "UTF-8"
))

# control how many loud LightGBM's logger is in tests
.LGB_VERBOSITY <- as.integer(
  Sys.getenv("LIGHTGBM_TEST_VERBOSITY", "-1")
)


# [description]
#    test that every element of 'x' is in 'y'
#
#    testthat::expect_in() was added in {testthat} v3.1.19.
#    This is here to support a similar interface on older {testthat} versions.
.expect_in <- function(x, y) {
  if (exists("expect_in")) {
    expect_in(x, y)
  } else {
    missing_items <- x[!(x %in% y)]
    if (length(missing_items) != 0L) {
      error_msg <- paste0("Some expected items not found: ", toString(missing_items))
      stop(error_msg)
    }
  }
}
