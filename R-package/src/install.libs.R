# User options
use_gpu <- FALSE
make_args_from_build_script <- character(0L)

# For Windows, the package will be built with Visual Studio
# unless you set one of these to TRUE
use_mingw <- FALSE
use_msys2 <- FALSE

if (use_mingw && use_msys2) {
  stop("Cannot use both MinGW and MSYS2. Please choose only one.")
}

if (.Machine$sizeof.pointer != 8L) {
  stop("LightGBM only supports 64-bit R, please check the version of R and Rtools.")
}

# Get some paths
source_dir <- file.path(R_PACKAGE_SOURCE, "src", fsep = "/")
build_dir <- file.path(source_dir, "build", fsep = "/")
inst_dir <- file.path(R_PACKAGE_SOURCE, "inst", fsep = "/")

# system() will not raise an R exception if the process called
# fails. Wrapping it here to get that behavior.
#
# system() introduces a lot of overhead, at least on Windows,
# so trying processx if it is available
.run_shell_command <- function(cmd, args, strict = TRUE) {
    on_windows <- .Platform$OS.type == "windows"
    has_processx <- suppressMessages({
      suppressWarnings({
        require("processx")  # nolint: undesirable_function, unused_import.
      })
    })
    if (has_processx && on_windows) {
      result <- processx::run(
        command = cmd
        , args = args
        , windows_verbatim_args = TRUE
        , error_on_status = FALSE
        , echo = TRUE
      )
      exit_code <- result$status
    } else {
      if (on_windows) {
        message(paste0(
          "Using system() to run shell commands. Installing "
          , "'processx' with install.packages('processx') might "
          , "make this faster."
        ))
      }
      cmd <- paste0(cmd, " ", paste(args, collapse = " "))
      exit_code <- system(cmd)
    }

    if (exit_code != 0L && isTRUE(strict)) {
        stop(paste0("Command failed with exit code: ", exit_code))
    }
    return(invisible(exit_code))
}

# try to generate Visual Studio build files
.generate_vs_makefiles <- function(cmake_args) {
  vs_versions <- c(
    "Visual Studio 17 2022"
    , "Visual Studio 16 2019"
    , "Visual Studio 15 2017"
  )
  working_vs_version <- NULL
  for (vs_version in vs_versions) {
    message(sprintf("Trying '%s'", vs_version))
    # if the build directory is not empty, clean it
    if (file.exists("CMakeCache.txt")) {
      file.remove("CMakeCache.txt")
    }
    vs_cmake_args <- c(
      cmake_args
      , "-G"
      , shQuote(vs_version)
      , "-A"
      , "x64"
    )
    exit_code <- .run_shell_command("cmake", c(vs_cmake_args, ".."), strict = FALSE)
    if (exit_code == 0L) {
      message(sprintf("Successfully created build files for '%s'", vs_version))
      return(invisible(TRUE))
    }

  }
  return(invisible(FALSE))
}

# Move in CMakeLists.txt
write_succeeded <- file.copy(
  file.path(inst_dir, "bin", "CMakeLists.txt")
  , "CMakeLists.txt"
  , overwrite = TRUE
)
if (!write_succeeded) {
  stop("Copying CMakeLists.txt failed")
}

# Prepare building package
dir.create(
  build_dir
  , recursive = TRUE
  , showWarnings = FALSE
)
setwd(build_dir)

use_visual_studio <- !(use_mingw || use_msys2)

# If using MSVC to build, pull in the script used
# to create R.def from R.dll
if (WINDOWS && use_visual_studio) {
  write_succeeded <- file.copy(
    file.path(inst_dir, "make-r-def.R")
    , file.path(build_dir, "make-r-def.R")
    , overwrite = TRUE
  )
  if (!write_succeeded) {
    stop("Copying make-r-def.R failed")
  }
}

# Prepare installation steps
cmake_args <- c(
  "-D__BUILD_FOR_R=ON"
  # pass in R version, to help FindLibR find the R library
  , sprintf("-DCMAKE_R_VERSION='%s.%s'", R.Version()[["major"]], R.Version()[["minor"]])
  # ensure CMake build respects how R is configured (`R CMD config SHLIB_EXT`)
  , sprintf("-DCMAKE_SHARED_LIBRARY_SUFFIX_CXX='%s'", SHLIB_EXT)
)
build_cmd <- "make"
build_args <- c("_lightgbm", make_args_from_build_script)
lib_folder <- file.path(source_dir, fsep = "/")

# add in command-line arguments
# NOTE: build_r.R replaces the line below
command_line_args <- NULL
cmake_args <- c(cmake_args, command_line_args)

WINDOWS_BUILD_TOOLS <- list(
  "MinGW" = c(
    build_tool = "mingw32-make.exe"
    , makefile_generator = "MinGW Makefiles"
  )
  , "MSYS2" = c(
    build_tool = "make.exe"
    , makefile_generator = "MSYS Makefiles"
  )
)

if (use_mingw) {
  windows_toolchain <- "MinGW"
} else if (use_msys2) {
  windows_toolchain <- "MSYS2"
} else {
  # Rtools 4.0 moved from MinGW to MSYS toolchain. If user tries
  # Visual Studio install but that fails, fall back to the toolchain
  # supported in Rtools
  windows_toolchain <- "MSYS2"
}
windows_build_tool <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["build_tool"]]
windows_makefile_generator <- WINDOWS_BUILD_TOOLS[[windows_toolchain]][["makefile_generator"]]

if (use_gpu) {
  cmake_args <- c(cmake_args, "-DUSE_GPU=ON")
}

# the checks below might already run `cmake -G`. If they do, set this flag
# to TRUE to avoid re-running it later
makefiles_already_generated <- FALSE

# Check if Windows installation (for gcc vs Visual Studio)
if (WINDOWS) {
  if (!use_visual_studio) {
    message(sprintf("Trying to build with %s", windows_toolchain))
    # Must build twice for Windows due sh.exe in Rtools
    cmake_args <- c(cmake_args, "-G", shQuote(windows_makefile_generator))
    .run_shell_command("cmake", c(cmake_args, ".."), strict = FALSE)
    build_cmd <- windows_build_tool
    build_args <- c("_lightgbm", make_args_from_build_script)
  } else {
    visual_studio_succeeded <- .generate_vs_makefiles(cmake_args)
    if (!isTRUE(visual_studio_succeeded)) {
      warning(sprintf("Building with Visual Studio failed. Attempting with %s", windows_toolchain))
      # Must build twice for Windows due sh.exe in Rtools
      cmake_args <- c(cmake_args, "-G", shQuote(windows_makefile_generator))
      .run_shell_command("cmake", c(cmake_args, ".."), strict = FALSE)
      build_cmd <- windows_build_tool
      build_args <- c("_lightgbm", make_args_from_build_script)
    } else {
      build_cmd <- "cmake"
      build_args <- c("--build", ".", "--target", "_lightgbm", "--config", "Release")
      lib_folder <- file.path(source_dir, "Release", fsep = "/")
      makefiles_already_generated <- TRUE
    }
  }
} else {
    .run_shell_command("cmake", c(cmake_args, ".."))
    makefiles_already_generated <- TRUE
}

# generate build files
if (!makefiles_already_generated) {
  .run_shell_command("cmake", c(cmake_args, ".."))
}

# build the library
message(paste0("Building lightgbm", SHLIB_EXT))
.run_shell_command(build_cmd, build_args)
src <- file.path(lib_folder, paste0("lightgbm", SHLIB_EXT), fsep = "/")

# Packages with install.libs.R need to copy some artifacts into the
# expected places in the package structure.
# see https://cran.r-project.org/doc/manuals/r-devel/R-exts.html#Package-subdirectories,
# especially the paragraph on install.libs.R
dest <- file.path(R_PACKAGE_DIR, paste0("libs", R_ARCH), fsep = "/")
dir.create(dest, recursive = TRUE, showWarnings = FALSE)
if (file.exists(src)) {
  message(paste0("Found library file: ", src, " to move to ", dest))
  file.copy(src, dest, overwrite = TRUE)

  symbols_file <- file.path(source_dir, "symbols.rds")
  if (file.exists(symbols_file)) {
    file.copy(symbols_file, dest, overwrite = TRUE)
  }

} else {
  stop(paste0("Cannot find lightgbm", SHLIB_EXT))
}

# clean up the "build" directory
if (dir.exists(build_dir)) {
  message("Removing 'build/' directory")
  unlink(
    x = build_dir
    , recursive = TRUE
    , force = TRUE
  )
}
