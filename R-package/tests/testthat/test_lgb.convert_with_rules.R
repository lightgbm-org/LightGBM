test_that("lgb.convert_with_rules() rejects inputs that are not a data.table or data.frame", {
    bad_inputs <- list(
        matrix(1.0:10.0, 2L, 5L)
        , TRUE
        , c("a", "b")
        , NA
        , 10L
        , lgb.Dataset(
            data = matrix(1.0:10.0, 2L, 5L)
            , params = list()
        )
    )
    for (bad_input in bad_inputs) {
        expect_error({
            conversion_result <- lgb.convert_with_rules(bad_input)
        }, regexp = "lgb.convert_with_rules: you provided", fixed = TRUE)
    }
})

test_that("lgb.convert_with_rules() should work correctly for a dataset with only character columns", {
    testDF <- data.frame(
        col1 = c("a", "b", "c")
        , col2 =  c("green", "green", "red")
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(input_data)
        # dataset should have been converted to integer
        converted_dataset <- conversion_result[["data"]]
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "integer")
        expect_identical(class(converted_dataset[["col2"]]), "integer")
        expect_identical(converted_dataset[["col1"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["col2"]], c(1L, 1L, 2L))
        # rules should be returned and correct
        rules <- conversion_result$rules
        expect_true(methods::is(rules, "list"))
        expect_length(rules, ncol(input_data))
        expect_identical(rules[["col1"]], c("a" = 1L, "b" = 2L, "c" = 3L))
        expect_identical(rules[["col2"]], c("green" = 1L, "red" = 2L))
    }
})

test_that("lgb.convert_with_rules() should work correctly for a dataset with only factor columns", {
    testDF <- data.frame(
        col1 = as.factor(c("a", "b", "c"))
        , col2 =  as.factor(c("green", "green", "red"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(input_data)
        # dataset should have been converted to integer
        converted_dataset <- conversion_result[["data"]]
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "integer")
        expect_identical(class(converted_dataset[["col2"]]), "integer")
        expect_identical(converted_dataset[["col1"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["col2"]], c(1L, 1L, 2L))
        # rules should be returned and correct
        rules <- conversion_result$rules
        expect_true(methods::is(rules, "list"))
        expect_length(rules, ncol(input_data))
        expect_identical(rules[["col1"]], c("a" = 1L, "b" = 2L, "c" = 3L))
        expect_identical(rules[["col2"]], c("green" = 1L, "red" = 2L))
    }
})

test_that("lgb.convert_with_rules() converts logical values by their rule names", {
    testDF <- data.frame(
        mixed = c(FALSE, TRUE, NA)
        , repeated = c(TRUE, TRUE, NA)
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(data.table::copy(input_data))
        expect_identical(conversion_result$data$mixed, c(0L, 1L, -1L))
        expect_identical(conversion_result$data$repeated, c(1L, 1L, -1L))
        expect_identical(conversion_result$rules$mixed, c("FALSE" = 0L, "TRUE" = 1L))

        custom_rules <- conversion_result$rules
        custom_rules$mixed <- c("TRUE" = 5L, "FALSE" = 2L)
        converted_data <- lgb.convert_with_rules(input_data, custom_rules)$data
        expect_identical(converted_data$mixed, c(2L, 5L, -1L))
    }
})

test_that("lgb.convert_with_rules() should not change a dataset with only integer columns", {
    testDF <- data.frame(
        col1 = 11L:15L
        , col2 = 16L:20L
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(input_data)
        # dataset should have been converted to integer
        converted_dataset <- conversion_result[["data"]]
        expect_identical(converted_dataset, input_data)
        # rules should be returned and correct
        rules <- conversion_result$rules
        expect_identical(rules, list())
    }
})

test_that("lgb.convert_with_rules() should work correctly for a dataset with numeric, factor, and character columns", {
    testDF <- data.frame(
        character_col = c("a", "b", "c")
        , numeric_col = c(1.0, 9.0, 10.0)
        , factor_col = as.factor(c("n", "n", "y"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(input_data)
        # dataset should have been converted to numeric
        converted_dataset <- conversion_result[["data"]]
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["character_col"]]), "integer")
        expect_identical(class(converted_dataset[["factor_col"]]), "integer")
        expect_identical(converted_dataset[["character_col"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["factor_col"]], c(1L, 1L, 2L))
        # rules should be returned and correct
        rules <- conversion_result$rules
        expect_true(methods::is(rules, "list"))
        expect_length(rules, 2L)
        expect_identical(rules[["character_col"]], c("a" = 1L, "b" = 2L, "c" = 3L))
        expect_identical(rules[["factor_col"]], c("n" = 1L, "y" = 2L))

        # today, lgb.convert_with_rules() does not convert numeric columns
        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, 10.0))
    }
})

test_that("lgb.convert_with_rules() should convert missing values to the expected value", {
    testDF <- data.frame(
        character_col = c("a", NA_character_, "c")
        , na_col = rep(NA, 3L)
        , na_real_col = rep(NA_real_, 3L)
        , na_int_col = rep(NA_integer_,  3L)
        , na_character_col = rep(NA_character_, 3L)
        , numeric_col = c(1.0, 9.0, NA_real_)
        , factor_col = as.factor(c("n", "n", "y"))
        , integer_col = c(1L, 9L, NA_integer_)
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        conversion_result <- lgb.convert_with_rules(input_data)
        # dataset should have been converted to integer
        converted_dataset <- conversion_result[["data"]]
        expect_identical(class(input_data), class(converted_dataset))

        expect_identical(class(converted_dataset[["character_col"]]), "integer")
        expect_identical(converted_dataset[["character_col"]], c(1L, 0L, 2L))

        # does not try to fill 0s in for already-integer columns
        expect_identical(class(converted_dataset[["integer_col"]]), "integer")
        expect_identical(converted_dataset[["integer_col"]], c(1L, 9L, NA_integer_))
        expect_identical(class(converted_dataset[["na_int_col"]]), "integer")
        expect_identical(converted_dataset[["na_int_col"]], rep(NA_integer_, nrow(converted_dataset)))

        expect_identical(class(converted_dataset[["factor_col"]]), "integer")
        expect_identical(converted_dataset[["factor_col"]], c(1L, 1L, 2L))

        # NAs in character columns should be converted to 0
        expect_identical(class(converted_dataset[["na_character_col"]]), "integer")
        expect_identical(converted_dataset[["na_character_col"]], rep(0L, nrow(converted_dataset)))

        # logical should be converted to integer
        expect_identical(class(converted_dataset[["na_col"]]), "integer")
        expect_identical(converted_dataset[["na_col"]], rep(-1L, 3L))

        # lgb.convert_with_rules() should not convert numeric columns to integer
        expect_identical(class(converted_dataset[["na_real_col"]]), "numeric")
        expect_identical(converted_dataset[["na_real_col"]], rep(NA_real_, nrow(converted_dataset)))
        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, NA_real_))

        # rules should be returned and correct
        rules <- conversion_result$rules
        expect_true(methods::is(rules, "list"))
        expect_length(rules, 3L)
        expect_identical(rules[["character_col"]], c("a" = 1L, "c" = 2L))
        expect_identical(rules[["factor_col"]], c("n" = 1L, "y" = 2L))
        expect_identical(rules[["na_col"]], stats::setNames(c(0L, 1L), c(FALSE, TRUE)))
    }
})

test_that("lgb.convert_with_rules() should work correctly if you provide your own well-formed rules", {
    testDF <- data.frame(
        character_col = c("a", NA_character_, "c", "a", "a", "c")
        , na_col = rep(NA, 6L)
        , na_real_col = rep(NA_real_, 6L)
        , na_int_col = rep(NA_integer_, 6L)
        , na_character_col = rep(NA_character_, 6L)
        , numeric_col = c(1.0, 9.0, NA_real_, 10.0, 11.0, 12.0)
        , factor_col = as.factor(c("n", "n", "y", "y", "n", "n"))
        , integer_col = c(1L, 9L, NA_integer_, 1L, 1L, 1L)
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)

    # value used by lgb.convert_with_rules() when it encounters a categorical value that
    # is not in the provided rules
    UNKNOWN_FACTOR_VALUE <- 0L
    UNKNOWN_LOGICAL_VALUE <- -1L
    for (input_data in list(testDF, testDT)) {
        custom_rules <- list(
            "character_col" = c(
                "a" = 5L
                , "c" = -10L
            )
            , "factor_col" = c(
                "n" = 65L
                , "y" = 66L
            )
        )
        conversion_result <- lgb.convert_with_rules(
            data = input_data
            , rules = custom_rules
        )

        # dataset should have been converted to integer
        converted_dataset <- conversion_result[["data"]]
        expect_identical(class(input_data), class(converted_dataset))

        expect_identical(class(converted_dataset[["character_col"]]), "integer")
        expect_identical(converted_dataset[["character_col"]], c(5L, UNKNOWN_FACTOR_VALUE, -10L, 5L, 5L, -10L))

        expect_identical(class(converted_dataset[["factor_col"]]), "integer")
        expect_identical(converted_dataset[["factor_col"]], c(65L, 65L, 66L, 66L, 65L, 65L))

        # columns not specified in rules are not going to be converted, unless they are all NA
        for (col in c("na_real_col", "na_int_col", "numeric_col", "integer_col")) {
            expect_identical(converted_dataset[[col]], input_data[[col]])
        }

        # non-numeric/integer columns that are all NA should have been filled in
        expect_identical(converted_dataset[["na_col"]], rep(UNKNOWN_LOGICAL_VALUE, 6L))
        expect_identical(converted_dataset[["na_character_col"]], rep(UNKNOWN_FACTOR_VALUE, 6L))

        # the rules you passed in should be returned unchanged
        rules <- conversion_result$rules
        expect_identical(rules, custom_rules)
    }
})

test_that("lgb.convert_with_rules() should modify data.tables in-place", {
    testDT <- data.table::data.table(
        character_col = c("a", NA_character_, "c")
        , na_col = rep(NA, 3L)
        , na_real_col = rep(NA_real_, 3L)
        , na_int_col = rep(NA_integer_,  3L)
        , na_character_col = rep(NA_character_, 3L)
        , numeric_col = c(1.0, 9.0, NA_real_)
        , factor_col = as.factor(c("n", "n", "y"))
        , integer_col = c(1L, 9L, NA_integer_)
    )
    conversion_result <- lgb.convert_with_rules(testDT)
    resultDT <- conversion_result[["data"]]
    expect_identical(resultDT, testDT)
})
