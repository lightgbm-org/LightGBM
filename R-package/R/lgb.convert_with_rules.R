# [description] get all column classes of a data.table or data.frame.
#               This function collapses the result of class() into a single string
.get_column_classes <- function(df) {
    return(
        vapply(
            X = df
            , FUN = function(x) {
                paste(class(x), collapse = ",")
            }
            , FUN.VALUE = character(1L)
        )
    )
}

# [description] check a data frame or data table for columns that are any
#               type other than numeric and integer. This is used by lgb.convert_with_rules()
#               to warn if more action is needed by users
#               before a dataset can be converted to a lgb.Dataset.
.warn_for_unconverted_columns <- function(df, function_name) {
    column_classes <- .get_column_classes(df = df)
    unconverted_columns <- column_classes[!(column_classes %in% c("numeric", "integer"))]
    if (length(unconverted_columns) > 0L) {
        col_detail_string <- toString(
            paste0(
                names(unconverted_columns)
                , " ("
                , unconverted_columns
                , ")"
            )
        )
        msg <- paste0(
            function_name
            , ": "
            , length(unconverted_columns)
            , " columns are not numeric or integer. These need to be dropped or converted to "
            , "be used in an lgb.Dataset object. "
            , col_detail_string
        )
        warning(msg)
    }
    return(invisible(NULL))
}

.LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA <- function() {
    return(-1L)
}
.LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA <- function() {
    return(0L)
}


#' @name lgb.convert_with_rules
#' @title Data preparator for LightGBM datasets with rules (integer)
#' @description Attempts to prepare a clean dataset to prepare to put in a \code{lgb.Dataset}.
#'              Factor, character, and logical columns are converted to integer. Missing values
#'              in factors and characters will be filled with 0L. Missing values in logicals
#'              will be filled with -1L.
#'
#'              This function returns and optionally takes in "rules" the describe exactly
#'              how to convert values in columns.
#'
#'              Columns that contain only NA values will be converted by this function but will
#'              not show up in the returned \code{rules}.
#'
#'              NOTE: In previous releases of LightGBM, this function was called \code{lgb.prepare_rules2}.
#' @param data A data.frame or data.table to prepare.
#' @param rules A set of rules from the data preparator, if already used. This should be an R list,
#'              where names are column names in \code{data} and values are named character
#'              vectors whose names are column values and whose values are new values to
#'              replace them with.
#' @return A list with the cleaned dataset (\code{data}) and the rules (\code{rules}).
#'         Note that the data must be converted to a matrix format (\code{as.matrix}) for input in
#'         \code{lgb.Dataset}.
#'
#' @examples
#' \donttest{
#' data(iris)
#'
#' str(iris)
#'
#' new_iris <- lgb.convert_with_rules(data = iris)
#' str(new_iris$data)
#'
#' data(iris) # Erase iris dataset
#' iris$Species[1L] <- "NEW FACTOR" # Introduce junk factor (NA)
#'
#' # Use conversion using known rules
#' # Unknown factors become 0, excellent for sparse datasets
#' newer_iris <- lgb.convert_with_rules(data = iris, rules = new_iris$rules)
#'
#' # Unknown factor is now zero, perfect for sparse datasets
#' newer_iris$data[1L, ] # Species became 0 as it is an unknown factor
#'
#' newer_iris$data[1L, 5L] <- 1.0 # Put back real initial value
#'
#' # Is the newly created dataset equal? YES!
#' all.equal(new_iris$data, newer_iris$data)
#'
#' # Can we test our own rules?
#' data(iris) # Erase iris dataset
#'
#' # We remapped values differently
#' personal_rules <- list(
#'   Species = c(
#'     "setosa" = 3L
#'     , "versicolor" = 2L
#'     , "virginica" = 1L
#'   )
#' )
#' newest_iris <- lgb.convert_with_rules(data = iris, rules = personal_rules)
#' str(newest_iris$data) # SUCCESS!
#' }
#' @importFrom data.table set
#' @export
lgb.convert_with_rules <- function(data, rules = NULL) {

    column_classes <- .get_column_classes(df = data)

    is_data_table <- data.table::is.data.table(x = data)
    is_data_frame <- is.data.frame(data)

    if (!(is_data_table || is_data_frame)) {
        stop(
            "lgb.convert_with_rules: you provided "
            , paste(class(data), collapse = " & ")
            , " but data should have class data.frame or data.table"
        )
    }

    # if user didn't provide rules, create them
    if (is.null(rules)) {
        rules <- list()
        columns_to_fix <- which(column_classes %in% c("character", "factor", "logical"))

        for (i in columns_to_fix) {

          col_values <- data[[i]]

          # Get unique values
          if (is.factor(col_values)) {
              unique_vals <- levels(col_values)
              unique_vals <- unique_vals[!is.na(unique_vals)]
              mini_numeric <- seq_along(unique_vals) # respect ordinal
          } else if (is.character(col_values)) {
              unique_vals <- as.factor(unique(col_values))
              unique_vals <- unique_vals[!is.na(unique_vals)]
              mini_numeric <- as.integer(unique_vals)  # no respect for ordinal
          } else if (is.logical(col_values)) {
              unique_vals <- c(FALSE, TRUE)
              mini_numeric <- c(0L, 1L)
          }

          # don't add rules for all-NA columns
          if (length(unique_vals) > 0L) {
              col_name <- names(data)[i]
              rules[[col_name]] <- mini_numeric
              names(rules[[col_name]]) <- unique_vals
          }
        }
    }

    for (col_name in names(rules)) {
        if (column_classes[[col_name]] == "logical") {
            default_value_for_na <- .LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA()
        } else {
            default_value_for_na <- .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()
        }
        if (is_data_table) {
            data.table::set(
                x = data
                , j = col_name
                , value = unname(rules[[col_name]][data[[col_name]]])
            )
            data[is.na(get(col_name)), (col_name) := default_value_for_na]
        } else {
            data[[col_name]] <- unname(rules[[col_name]][data[[col_name]]])
            data[is.na(data[col_name]), col_name] <- default_value_for_na
        }
    }

    # if any all-NA columns exist, they won't be in rules. Convert them
    all_na_cols <- which(
        sapply(
            X = data
            , FUN = function(x) {
                (is.factor(x) || is.character(x) || is.logical(x)) && all(is.na(unique(x)))
            }
        )
    )
    for (col_name in all_na_cols) {
        if (column_classes[[col_name]] == "logical") {
            default_value_for_na <- .LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA()
        } else {
            default_value_for_na <- .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()
        }
        if (is_data_table) {
            data[, (col_name) := rep(default_value_for_na, .N)]
        } else {
            data[[col_name]] <- default_value_for_na
        }
    }

    .warn_for_unconverted_columns(df = data, function_name = "lgb.convert_with_rules")

    return(list(data = data, rules = rules))

}
