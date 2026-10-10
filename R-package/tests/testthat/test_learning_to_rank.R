test_that("learning-to-rank with lgb.train() works as expected", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    # just keep a few features,to generate an model with imperfect fit
    train <- agaricus.train
    train_data <- train$data[1L:6000L, 1L:20L]
    dtrain <- lgb.Dataset(
        train_data
        , label = train$label[1L:6000L]
        , group = rep(150L, 40L)
    )
    ndcg_at <- "1,2,3"
    eval_names <- paste0("ndcg@", strsplit(ndcg_at, ",", fixed = TRUE)[[1L]])
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = ndcg_at
        , lambdarank_truncation_level = 3L
        , learning_rate = 0.001
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 10L
    )
    expect_true(.is_Booster(model))

    dumped_model <- jsonlite::fromJSON(
        model$dump_model()
    )
    expect_equal(dumped_model[["objective"]], "lambdarank")
    expect_equal(dumped_model[["max_feature_idx"]], ncol(train_data) - 1L)

    # check that evaluation results make sense (0.0 < nDCG < 1.0)
    eval_results <- model$eval_train()
    expect_equal(length(eval_results), length(eval_names))
    for (result in eval_results) {
        expect_true(result[["value"]] > 0.0)
        expect_true(result[["value"]] < 1.0)
        expect_true(result[["higher_better"]])
        expect_identical(result[["data_name"]], "training")
    }
    expect_identical(
        sapply(
            X = eval_results
            , FUN = function(x) {
                x$name
            }
        )
        , eval_names
    )
    expect_equal(eval_results[[1L]][["value"]], 0.775)
    if (!.LGB_ON_32_BIT_WINDOWS) {
        expect_true(abs(eval_results[[2L]][["value"]] - 0.745986) < .LGB_NUMERIC_TOLERANCE)
        expect_true(abs(eval_results[[3L]][["value"]] - 0.7351959) < .LGB_NUMERIC_TOLERANCE)
    }
})

test_that("learning-to-rank with lgb.cv() works as expected", {
    testthat::skip_if(
        .LGB_ON_32_BIT_WINDOWS
        , message = "Skipping on 32-bit Windows"
    )
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    # just keep a few features,to generate an model with imperfect fit
    train <- agaricus.train
    train_data <- train$data[1L:6000L, 1L:20L]
    dtrain <- lgb.Dataset(
        train_data
        , label = train$label[1L:6000L]
        , group = rep(150L, 40L)
    )
    ndcg_at <- "1,2,3"
    eval_names <- paste0("ndcg@", strsplit(ndcg_at, ",", fixed = TRUE)[[1L]])
    params <- list(
        objective = "lambdarank"
        , metric = "ndcg"
        , ndcg_at = ndcg_at
        , lambdarank_truncation_level = 3L
        , label_gain = "0,1,3"
        , min_data = 1L
        , learning_rate = 0.01
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    nfold <- 4L
    nrounds <- 10L
    cv_bst <- lgb.cv(
        params = params
        , data = dtrain
        , nrounds = nrounds
        , nfold = nfold
    )
    expect_true(methods::is(cv_bst, "lgb.CVBooster"))
    expect_equal(length(cv_bst$boosters), nfold)

    # "valid" should contain results for each metric
    eval_results <- cv_bst$record_evals[["valid"]]
    eval_names <- c("ndcg@1", "ndcg@2", "ndcg@3")
    expect_identical(names(eval_results), eval_names)

    # check that best score and iter make sense (0.0 < nDCG < 1.0)
    best_iter <- cv_bst$best_iter
    best_score <- cv_bst$best_score
    expect_true(best_iter > 0L)
    expect_true(best_iter <= nrounds)
    expect_true(best_score > 0.0)
    expect_true(best_score < 1.0)
    expect_true(abs(best_score - 0.75) < .LGB_NUMERIC_TOLERANCE)

    # best_score should be set for the first metric
    first_metric <- eval_names[[1L]]
    expect_equal(best_score, eval_results[[first_metric]][["eval"]][[best_iter]])

    for (eval_name in eval_names) {
        results_for_this_metric <- eval_results[[eval_name]]

        # each set of metrics should have eval and eval_err
        expect_identical(names(results_for_this_metric), c("eval", "eval_err"))

        # there should be one "eval" and "eval_err" per round
        expect_equal(length(results_for_this_metric[["eval"]]), nrounds)
        expect_equal(length(results_for_this_metric[["eval_err"]]), nrounds)

        # check that evaluation results make sense (0.0 < nDCG < 1.0)
        all_evals <- unlist(results_for_this_metric[["eval"]])
        expect_true(all(all_evals > 0.0 & all_evals < 1.0))
    }

    # first and last value of each metric should be as expected
    ndcg1_values <- c(0.675, 0.725, 0.65, 0.725, 0.75, 0.725, 0.75, 0.725, 0.75, 0.75)
    expect_true(all(abs(unlist(eval_results[["ndcg@1"]][["eval"]]) - ndcg1_values) < .LGB_NUMERIC_TOLERANCE))

    ndcg2_values <- c(
        0.6556574, 0.6669721, 0.6306574, 0.6476294, 0.6629581,
        0.6476294, 0.6629581, 0.6379581, 0.7113147, 0.6823008
    )
    expect_true(all(abs(unlist(eval_results[["ndcg@2"]][["eval"]]) - ndcg2_values) < .LGB_NUMERIC_TOLERANCE))

    ndcg3_values <- c(
        0.6484639, 0.6571238, 0.6469279, 0.6540516, 0.6481857,
        0.6481857, 0.6481857, 0.6466496, 0.7027939, 0.6629898
    )
    expect_true(all(abs(unlist(eval_results[["ndcg@3"]][["eval"]]) - ndcg3_values) < .LGB_NUMERIC_TOLERANCE))

    # check details of each booster
    for (bst in cv_bst$boosters) {
        dumped_model <- jsonlite::fromJSON(
            bst$booster$dump_model()
        )
        expect_equal(dumped_model[["objective"]], "lambdarank")
        expect_equal(dumped_model[["max_feature_idx"]], ncol(train_data) - 1L)
    }
})
