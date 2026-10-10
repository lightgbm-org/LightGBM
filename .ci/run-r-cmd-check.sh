#!/bin/bash

set -e -u -o pipefail

PKG_TARBALL="${1}"
declare -i ALLOWED_CHECK_NOTES=${2}

# 'R CMD check' redirects installation logs to a file, and returns
# a non-0 exit code if ERRORs are raised.
#
# The '||' here gives us an opportunity to echo out the installation
# logs prior to exiting the script.
check_succeeded="yes"
R CMD check "${PKG_TARBALL}" \
    --as-cran \
    --run-donttest \
|| check_succeeded="no"

CHECK_LOG_FILE=lightgbm.Rcheck/00check.log
BUILD_LOG_FILE=lightgbm.Rcheck/00install.out

echo "R CMD check build logs:"
cat "${BUILD_LOG_FILE}"

if [[ $check_succeeded == "no" ]]; then
    echo "R CMD check failed"
    exit 1
fi

# WARNINGs or ERRORs should be treated as a failure
if grep -q -E "WARNING|ERROR" "${CHECK_LOG_FILE}"; then
    echo "WARNINGs or ERRORs have been found by R CMD check"
    exit 1
fi

# Allow a configurable number of NOTEs.
# Sometimes NOTEs are raised in CI that wouldn't show up on an actual CRAN submission.
set +e
NUM_CHECK_NOTES=$(
    grep -o -E '[0-9]+ NOTE' "${CHECK_LOG_FILE}" \
    | sed 's/[^0-9]*//g'
)
if [[ ${NUM_CHECK_NOTES} -gt ${ALLOWED_CHECK_NOTES} ]]; then
    echo "Found ${NUM_CHECK_NOTES} NOTEs from R CMD check. Only ${ALLOWED_CHECK_NOTES} are allowed"
    exit 1
fi
