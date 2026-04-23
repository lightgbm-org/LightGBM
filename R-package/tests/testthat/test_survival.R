.make_survival <- function(n_samples = 500L, n_features = 5L, random_state = 0L) {
    set.seed(random_state)
    X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
    log_hazard <- X[, 1L] + 0.5 * X[, 2L]
    times <- rexp(n_samples, rate = exp(log_hazard))
    censoring_rate <- 0.3
    censor_times <- rexp(n_samples, rate = censoring_rate / median(times))
    y <- pmin(times, censor_times)
    censored <- censor_times < times
    y[censored] <- -y[censored]
    list(X = X, y = y)
}

test_that("survival_cox with lgb.train() works as expected", {
    surv <- .make_survival()
    n_train <- 375L
    n <- nrow(surv$X)
    dtrain <- lgb.Dataset(surv$X[1L:n_train, ], label = surv$y[1L:n_train])
    dval <- lgb.Dataset(
        surv$X[(n_train + 1L):n, ]
        , label = surv$y[(n_train + 1L):n]
        , reference = dtrain
    )

    params <- list(
        objective = "survival_cox"
        , metric = list("survival_cox_nll", "concordance_index")
        , num_leaves = 8L
        , seed = 708L
        , num_threads = .LGB_MAX_THREADS
        , deterministic = TRUE
        , force_row_wise = TRUE
        , verbose = .LGB_VERBOSITY
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 10L
        , valids = list(val = dval)
        , record = TRUE
    )

    # check that both metrics are present in expected order
    eval_results <- model$eval_valid()
    expect_equal(length(eval_results), 2L)
    expect_equal(eval_results[[1L]]$name, "survival_cox_nll")
    expect_equal(eval_results[[2L]]$name, "concordance_index")

    # check higher_better flags
    expect_false(eval_results[[1L]]$higher_better)
    expect_true(eval_results[[2L]]$higher_better)

    # extract per-round metric values
    losses <- unlist(model$record_evals[["val"]][["survival_cox_nll"]][["eval"]])
    c_indices <- unlist(model$record_evals[["val"]][["concordance_index"]][["eval"]])
    expect_equal(length(losses), 10L)
    expect_equal(length(c_indices), 10L)

    # check that all metrics are finite
    expect_true(all(is.finite(losses)))
    expect_true(all(is.finite(c_indices)))

    # check that metrics are in a reasonable range for this problem
    expect_true(all(losses > 3.7 & losses < 4.1))
    expect_true(all(c_indices > 0.6 & c_indices < 0.8))

    # check that validation loss generally improves (last < first)
    expect_true(losses[1L] > losses[10L])

    # check that concordance index and loss improves for at least half the rounds
    loss_improvements <- sum(diff(losses) < 0L)
    ci_improvements <- sum(diff(c_indices) > 0L)
    expect_true(loss_improvements >= 5L)
    expect_true(ci_improvements >= 5L)
})
