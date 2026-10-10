test_that(".params2str() works as expected for empty lists", {
    out_str <- .params2str(
        params = list()
    )
    expect_identical(class(out_str), "character")
    expect_equal(out_str, "")
})

test_that(".params2str() works as expected for a key in params with multiple different-length elements", {
    metrics <- c("a", "ab", "abc", "abcdefg")
    params <- list(
        objective = "magic"
        , metric = metrics
        , nrounds = 10L
        , learning_rate = 0.0000001
    )
    out_str <- .params2str(
        params = params
    )
    expect_identical(class(out_str), "character")
    expect_identical(
        out_str
        , "objective=magic metric=a,ab,abc,abcdefg nrounds=10 learning_rate=0.0000001"
    )
})

test_that(".params2str() passes through duplicated params", {
    out_str <- .params2str(
        params = list(
            objective = "regression"
            , bagging_fraction = 0.8
            , bagging_fraction = 0.5  # nolint: duplicate_argument.
        )
    )
    expect_equal(out_str, "objective=regression bagging_fraction=0.8 bagging_fraction=0.5")
})

test_that(".check_eval works as expected with no metric", {
    params <- .check_eval(
        params = list(device = "cpu")
        , eval = "binary_error"
    )
    expect_named(params, c("device", "metric"))
    expect_identical(params[["metric"]], list("binary_error"))
})

test_that(".check_eval adds eval to metric in params", {
    params <- .check_eval(
        params = list(metric = "auc")
        , eval = "binary_error"
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error"))
})

test_that(".check_eval adds eval to metric in params if two evaluation names are provided", {
    params <- .check_eval(
        params = list(metric = "auc")
        , eval = c("binary_error", "binary_logloss")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error", "binary_logloss"))
})

test_that(".check_eval adds eval to metric in params if a list is provided", {
    params <- .check_eval(
        params = list(metric = "auc")
        , eval = list("binary_error", "binary_logloss")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("auc", "binary_error", "binary_logloss"))
})

test_that(".check_eval drops duplicate metrics and preserves order", {
    params <- .check_eval(
        params = list(metric = "l1")
        , eval = list("l2", "rmse", "l1", "rmse")
    )
    expect_named(params, "metric")
    expect_identical(params[["metric"]], list("l1", "l2", "rmse"))
})

test_that(".check_wrapper_param() uses passed-in keyword arg if no alias found in params", {
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- .check_wrapper_param(
        main_param_name = "num_iterations"
        , params = list()
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], kwarg_val)
})

test_that(".check_wrapper_param() prefers main parameter to alias and keyword arg", {
    num_iterations <- sample(seq_len(100L), size = 1L)
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- .check_wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            num_iterations = num_iterations
            , num_tree = sample(seq_len(100L), size = 1L)
            , n_estimators = sample(seq_len(100L), size = 1L)
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], num_iterations)

    # aliases should be removed
    expect_identical(params, list(num_iterations = num_iterations))
})

test_that(".check_wrapper_param() prefers alias to keyword arg", {
    n_estimators <- sample(seq_len(100L), size = 1L)
    num_tree <- sample(seq_len(100L), size = 1L)
    kwarg_val <- sample(seq_len(100L), size = 1L)
    params <- .check_wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            num_tree = num_tree
            , n_estimators = n_estimators
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params[["num_iterations"]], num_tree)
    expect_identical(params, list(num_iterations = num_tree))

    # switching the order shouldn't switch which one is chosen
    params2 <- .check_wrapper_param(
        main_param_name = "num_iterations"
        , params = list(
            n_estimators = n_estimators
            , num_tree = num_tree
        )
        , alternative_kwarg_value = kwarg_val
    )
    expect_equal(params2[["num_iterations"]], num_tree)
    expect_identical(params2, list(num_iterations = num_tree))
})

test_that(".equal_or_both_null produces expected results", {
    expect_true(.equal_or_both_null(NULL, NULL))
    expect_false(.equal_or_both_null(1.0, NULL))
    expect_false(.equal_or_both_null(NULL, 1.0))
    expect_true(.equal_or_both_null(1.0, 1.0))
    expect_true(.equal_or_both_null(1.0, 1L))
    expect_false(.equal_or_both_null(NA, NULL))
    expect_false(.equal_or_both_null(NULL, NA))
    expect_false(.equal_or_both_null(10.0, 1L))
    expect_true(.equal_or_both_null(0L, 0L))
})

test_that(".check_interaction_constraints() adds skipped features", {
  ref <- letters[1L:5L]
  ic_num <- list(1L, c(2L, 3L))
  ic_char <- list("a", c("b", "c"))
  expected <- list("[0]", "[1,2]", "[3,4]")

  ic_checked_num <- .check_interaction_constraints(
    interaction_constraints = ic_num, column_names = ref
  )

  ic_checked_char <- .check_interaction_constraints(
    interaction_constraints = ic_char, column_names = ref
  )

  expect_equal(ic_checked_num, expected)
  expect_equal(ic_checked_char, expected)
})
