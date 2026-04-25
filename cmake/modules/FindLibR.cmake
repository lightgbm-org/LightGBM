# CMake module used to find the location of R's
# dll and header files.
#
# Borrows heavily from xgboost's R package:
#
# * https://github.com/dmlc/xgboost/blob/master/cmake/modules/FindLibR.cmake
#
# Defines the following:
#  LIBR_FOUND
#  LIBR_HOME
#  LIBR_EXECUTABLE
#  LIBR_MSVC_CORE_LIBRARY
#  LIBR_INCLUDE_DIRS
#  LIBR_LIBS_DIR
#  LIBR_CORE_LIBRARY
# and a CMake function to create R.lib for MSVC

# lint_cmake: -convention/filename

if(NOT R_ARCH)
  if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
    set(R_ARCH "i386")
  else()
    set(R_ARCH "x64")
  endif()
endif()

if(NOT ("${R_ARCH}" STREQUAL "x64"))
  message(FATAL_ERROR "LightGBM's R-package currently only supports 64-bit operating systems")
endif()

# Creates R.lib and R.def in the build directory for linking with MSVC
# https://docs.microsoft.com/en-us/cpp/build/reference/link-input-files?redirectedfrom=MSDN&view=vs-2019
function(create_rlib_for_msvc)

  message(STATUS "Creating R.lib and R.def")

  # various checks and warnings
  if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "create_rlib_for_msvc() can only be used with MSVC")
  endif()

  if(NOT EXISTS "${LIBR_CORE_LIBRARY}")
    message(FATAL_ERROR "LIBR_CORE_LIBRARY, '${LIBR_CORE_LIBRARY}', not found")
  endif()

  find_program(DLLTOOL_EXE dlltool)

  if(NOT DLLTOOL_EXE)
    message(FATAL_ERROR "dlltool.exe not found!\nDo you have Rtools installed with its MinGW's bin/ in PATH?")
  endif()

  set(LIBR_MSVC_CORE_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/R.lib" CACHE PATH "R.lib filepath")

  get_filename_component(
    LIBR_RSCRIPT_EXECUTABLE_DIR
    ${LIBR_EXECUTABLE}
    DIRECTORY
  )
  set(LIBR_RSCRIPT_EXECUTABLE "${LIBR_RSCRIPT_EXECUTABLE_DIR}/Rscript")

  execute_process(
    COMMAND ${LIBR_RSCRIPT_EXECUTABLE}
    "${CMAKE_CURRENT_BINARY_DIR}/make-r-def.R"
    "${LIBR_CORE_LIBRARY}" "${CMAKE_CURRENT_BINARY_DIR}/R.def"
  )
  execute_process(
    COMMAND ${DLLTOOL_EXE}
    "--input-def" "${CMAKE_CURRENT_BINARY_DIR}/R.def"
    "--output-lib" "${LIBR_MSVC_CORE_LIBRARY}"
  )
endfunction()

# R version information is used to search for R's libraries in
# the registry on Windows. Since this code is orchestrated by
# an R script (src/install.libs.R), that script uses R's built-ins to
# find the version of R and pass it through as a CMake variable
if(CMAKE_R_VERSION)
  message(STATUS "R version passed into FindLibR.cmake: ${CMAKE_R_VERSION}")
elseif(WIN32)
  message(
    FATAL_ERROR
    "Expected CMAKE_R_VERSION to be passed in on Windows but none was provided. Check src/install.libs.R"
  )
endif()


if(NOT LIBR_EXECUTABLE)
  find_program(
    LIBR_EXECUTABLE
    NAMES R R.exe
  )

  # CRAN may run RD CMD CHECK instead of R CMD CHECK,
  # which can lead to this infamous error:
  # 'R' should not be used without a path -- see par. 1.6 of the manual
  if(LIBR_EXECUTABLE MATCHES ".*\\.Rcheck.*")
    unset(LIBR_EXECUTABLE CACHE)
  endif()

  # ignore the R bundled with R.app on Mac, since that is GUI-only
  if(LIBR_EXECUTABLE MATCHES ".+R\\.app.*")
    unset(LIBR_EXECUTABLE CACHE)
  endif()
endif()

# Find R executable unless it has been provided directly or already found
if(NOT LIBR_EXECUTABLE)
  if(APPLE)

    find_library(LIBR_LIBRARIES R)

    if(LIBR_LIBRARIES MATCHES ".*\\.framework")
      set(LIBR_HOME "${LIBR_LIBRARIES}/Resources")
      set(LIBR_EXECUTABLE "${LIBR_HOME}/R")
    else()
      get_filename_component(_LIBR_LIBRARIES "${LIBR_LIBRARIES}" REALPATH)
      get_filename_component(_LIBR_LIBRARIES_DIR "${_LIBR_LIBRARIES}" DIRECTORY)
      set(LIBR_EXECUTABLE "${_LIBR_LIBRARIES_DIR}/../bin/R")
    endif()

  elseif(UNIX)

    # attempt to find R executable
    if(NOT LIBR_EXECUTABLE)
      find_program(
        LIBR_EXECUTABLE
        NO_DEFAULT_PATH
        HINTS "${CMAKE_CURRENT_BINARY_DIR}" "/usr/bin" "/usr/lib/" "/usr/local/bin/"
        NAMES R
      )
    endif()

  # Windows
  else()

    # if R executable not available, query R_HOME path from registry
    if(NOT LIBR_HOME)

      # Try to find R's location in the registry
      # ref: https://cran.r-project.org/bin/windows/base/rw-FAQ.html#Does-R-use-the-Registry_003f
      get_filename_component(
        LIBR_HOME
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\R-core\\R\\${CMAKE_R_VERSION};InstallPath]"
        ABSOLUTE
      )
    endif()

    if(NOT LIBR_HOME)
      get_filename_component(
        LIBR_HOME
        "[HKEY_CURRENT_USER\\SOFTWARE\\R-core\\R\\${CMAKE_R_VERSION};InstallPath]"
        ABSOLUTE
      )
    endif()

    if(NOT LIBR_HOME)
      message(
        FATAL_ERROR
        "Unable to locate R executable.\
\nEither add its location to PATH or provide it through the LIBR_EXECUTABLE CMake variable"
      )
    endif()

    # set exe location based on R_ARCH
    set(LIBR_EXECUTABLE "${LIBR_HOME}/bin/${R_ARCH}/R.exe")

  endif()

  if(NOT LIBR_EXECUTABLE)
    message(
      FATAL_ERROR
      "Unable to locate R executable.\
\nEither add its location to PATH or provide it through the LIBR_EXECUTABLE CMake variable"
    )
  endif()

endif()

# ask R for the home path
execute_process(
  COMMAND ${LIBR_EXECUTABLE} "--slave" "--vanilla" "-e" "cat(normalizePath(R.home(), winslash='/'))"
  OUTPUT_VARIABLE LIBR_HOME
)

# ask R for the include dir
execute_process(
  COMMAND ${LIBR_EXECUTABLE} "--slave" "--vanilla" "-e" "cat(normalizePath(R.home('include'), winslash='/'))"
  OUTPUT_VARIABLE LIBR_INCLUDE_DIRS
)

# ask R for the lib dir
execute_process(
  COMMAND ${LIBR_EXECUTABLE} "--slave" "--vanilla" "-e" "cat(normalizePath(R.home('lib'), winslash='/'))"
  OUTPUT_VARIABLE LIBR_LIBS_DIR
)

set(LIBR_HOME ${LIBR_HOME} CACHE PATH "R home directory")
set(LIBR_EXECUTABLE ${LIBR_EXECUTABLE} CACHE PATH "R executable")
set(LIBR_INCLUDE_DIRS ${LIBR_INCLUDE_DIRS} CACHE PATH "R include directory")
set(LIBR_LIBS_DIR ${LIBR_LIBS_DIR} CACHE PATH "Where R stores vendored third-party libraries")

# where is R.so / R.dll / libR.so likely to be found?
set(
  LIBR_PATH_HINTS
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${LIBR_HOME}/lib"
    "${LIBR_HOME}/bin/${R_ARCH}"
    "${LIBR_HOME}/bin"
    "${LIBR_LIBRARIES}"
)

# look for the core R library
find_library(
  LIBR_CORE_LIBRARY
  NAMES R R.dll
  HINTS ${LIBR_PATH_HINTS}
)

# starting from CMake 3.17, find_library() will not find .dll files by default
# https://cmake.org/cmake/help/v3.17/release/3.17.html#other-changes
if(WIN32 AND NOT LIBR_CORE_LIBRARY)
  find_file(
    LIBR_CORE_LIBRARY
    NAME R.dll
    HINTS ${LIBR_PATH_HINTS}
  )
endif()

set(LIBR_CORE_LIBRARY ${LIBR_CORE_LIBRARY} CACHE PATH "R core shared library")

if(WIN32 AND MSVC)

  # create a local R.lib import library for R.dll if it doesn't exist
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/R.lib")
    create_rlib_for_msvc()
  endif()

endif()

# define find requirements
include(FindPackageHandleStandardArgs)

if(WIN32 AND MSVC)
# lint_cmake: -package/stdargs
  find_package_handle_standard_args(
# lint_cmake: +package/stdargs
    LibR DEFAULT_MSG
    LIBR_HOME
    LIBR_EXECUTABLE
    LIBR_INCLUDE_DIRS
    LIBR_LIBS_DIR
    LIBR_CORE_LIBRARY
    LIBR_MSVC_CORE_LIBRARY
  )
else()
# lint_cmake: -package/stdargs
  find_package_handle_standard_args(
# lint_cmake: +package/stdargs
    LibR DEFAULT_MSG
    LIBR_HOME
    LIBR_EXECUTABLE
    LIBR_INCLUDE_DIRS
    LIBR_LIBS_DIR
    LIBR_CORE_LIBRARY
  )
endif()
