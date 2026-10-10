# Install R dependencies, using only base R.
#
# Supported arguments:
#
#  --all                     Install all the 'Depends', 'Imports', 'LinkingTo', and 'Suggests' dependencies
#                            (automatically implies --build --test).
#
#  --build                   Install the packages needed to build.
#
#  --exclude=<pkg1,pkg2,...> Comma-delimited list of packages to NOT install.
#
#  --include=<pkg1,pkg2,...> Comma-delimited list of additional packages to install.
#                            These will always be installed, unless also used in "--exclude".
#
#  --test                    Install packages needed to run tests.
#


# [description] Parse command line arguments into an R list.
#               Returns a list where keys are arguments and values
#               are either TRUE (for flags) or a vector of values passed via a
#               comma-delimited list.
.parse_args <- function(args) {
    out <- list(
        "--all"       = FALSE
        , "--build"   = FALSE
        , "--exclude" = character(0L)
        , "--include" = character(0L)
        , "--test"    = FALSE
    )
    for (arg in args) {
        parsed_arg <- unlist(strsplit(arg, "=", fixed = TRUE))
        arg_name <- parsed_arg[[1L]]
        if (!(arg_name %in% names(out))) {
            stop(sprintf("Unrecognized argument: '%s'", arg_name))
        }
        if (length(parsed_arg) == 2L) {
            # lists, like "--include=roxygen2,testthat"
            values <- unlist(strsplit(parsed_arg[[2L]], ",", fixed = TRUE))
            out[[arg_name]] <- values
        } else {
            # flags, like "--build"
            out[[arg]] <- TRUE
        }
    }
    return(out)
}

args <- .parse_args(
    commandArgs(trailingOnly = TRUE)
)

# which dependencies to install
ALL_DEPS     <- isTRUE(args[["--all"]])
BUILD_DEPS   <- ALL_DEPS || isTRUE(args[["--build"]])
TEST_DEPS    <- ALL_DEPS || isTRUE(args[["--test"]])

# force downloading of binary packages on macOS
COMPILE_FROM_SOURCE <- "both"
PACKAGE_TYPE <- getOption("pkgType")

# CRAN has precompiled binaries for macOS and Windows... prefer those,
# for faster installation.
if (Sys.info()[["sysname"]] == "Darwin" || .Platform$OS.type == "windows") {
    COMPILE_FROM_SOURCE <- "never"
    PACKAGE_TYPE <- "binary"
}
options(
    install.packages.check.source = "no"
    , install.packages.compile.from.source = COMPILE_FROM_SOURCE
)

# always use the same CRAN mirror
CRAN_MIRROR <- Sys.getenv("CRAN_MIRROR", unset = "https://cran.r-project.org")

# we always want these
deps_to_install <- c(
    "data.table"
    , "jsonlite"
    , "Matrix"
    , "R6"
)

if (isTRUE(BUILD_DEPS)) {
    deps_to_install <- c(
        deps_to_install
        , "knitr"
        , "markdown"
    )
}

if (isTRUE(TEST_DEPS)) {
    deps_to_install <- c(
        deps_to_install
        , "RhpcBLASctl"
        , "testthat"
    )
}

# add packages passed through '--include'
deps_to_install <- unique(c(
    deps_to_install
    , args[["--include"]]
))

# remove packages passed through '--exclude'
deps_to_install <- setdiff(
    x = deps_to_install
    , args[["--exclude"]]
)

msg <- sprintf(
    "[install-r-deps] installing R packages: %s\n"
    , toString(sort(deps_to_install))
)
cat(msg)

install.packages(  # nolint: undesirable_function.
    pkgs = deps_to_install
    , dependencies = c("Depends", "Imports", "LinkingTo")
    , lib = Sys.getenv("R_LIB_PATH", unset = .libPaths()[[1L]])
    , repos = CRAN_MIRROR
    , type = PACKAGE_TYPE
    , Ncpus = parallel::detectCores()
)
