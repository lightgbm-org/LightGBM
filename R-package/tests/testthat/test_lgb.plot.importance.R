test_that("lgb.plot.importance() should run without error for well-formed inputs", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    params <- list(
        objective = "binary"
        , learning_rate = 0.01
        , num_leaves = 63L
        , max_depth = -1L
        , min_data_in_leaf = 1L
        , min_sum_hessian_in_leaf = 1.0
        , verbosity = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    model <- lgb.train(params, dtrain, 3L)
    tree_imp <- lgb.importance(model, percentage = TRUE)

    # Check that there are no plots present before plotting
    expect_null(dev.list())

    args_no_cex <- list(
        "tree_imp" = tree_imp
        , top_n = 10L
        , measure = "Gain"
    )
    args_cex <- args_no_cex
    args_cex[["cex"]] <- 0.75

    for (arg_list in list(args_no_cex, args_cex)) {

        resDT <- do.call(
            what = lgb.plot.importance
            , args = arg_list
        )

        # Check that lgb.plot.importance() returns the data.table of the plotted data
        expect_true(data.table::is.data.table(resDT))
        expect_named(resDT, c("Feature", "Gain", "Cover", "Frequency"))

        # Check that a plot was produced
        expect_false(is.null(dev.list()))

        # remove all plots
        dev.off()
        expect_null(dev.list())
    }
})
