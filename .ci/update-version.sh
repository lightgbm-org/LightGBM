#!/bin/sh
#
# [description]
#     Update files in source control based on the content of 'VERSION.txt'.
#
# [usage]
#
#     update-version.sh

set -e -u

LGB_VERSION=$(head -1 ./VERSION.txt)
LGB_VERSION_NO_RC=$(echo "${LGB_VERSION}" | sed 's/rc/-/g')

# in-place 'sed' that's compatible with GNU sed and BSD sed (the one bundled with macOS).
# Writes through a temporary file (instead of command substitution) to preserve trailing
# newlines, and overwrites with 'cat' (instead of 'mv') to preserve file permissions.
update_file() {
    TARGET_FILE=$1
    SED_EXPRESSION=$2
    TMP_FILE=$(mktemp)
    sed "${SED_EXPRESSION}" "${TARGET_FILE}" > "${TMP_FILE}"
    cat "${TMP_FILE}" > "${TARGET_FILE}"
    rm "${TMP_FILE}"
}

update_file \
    ./.appveyor.yml \
    "s|^version: .*$|version: ${LGB_VERSION}.{build}|"

update_file \
    ./python-package/pyproject.toml \
    "s|^version = \"[0-9a-z.]\{1,\}\"$|version = \"${LGB_VERSION}\"|"

# R packages cannot have versions like 3.0.0rc1, but 3.0.0-1 is acceptable
update_file \
    ./R-package/DESCRIPTION \
    "s|^Version: .*$|Version: ${LGB_VERSION_NO_RC}|"

update_file \
    ./R-package/configure.ac \
    "s|^AC_INIT.*$|AC_INIT([lightgbm], [${LGB_VERSION_NO_RC}], [], [lightgbm], [])|"

CONFIGURE_VERSION=$(sed -n "s|^PACKAGE_VERSION='\(.*\)'$|\1|p" ./R-package/configure)
CONFIGURE_VERSION_PATTERN=$(echo "${CONFIGURE_VERSION}" | sed 's|\.|\\.|g')
update_file \
    ./R-package/configure \
    "s|${CONFIGURE_VERSION_PATTERN}|${LGB_VERSION_NO_RC}|g"

# only mark the R docs as released on release branches (like 'release/4.8.0')
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2> /dev/null || true)
if echo "${GIT_BRANCH}" | grep -q -E '^release/v[0-9]+\.'; then
    PKGDOWN_MODE="release"
else
    PKGDOWN_MODE="unreleased"
fi

update_file \
    ./R-package/pkgdown/_pkgdown.yml \
    "s|^  mode: .*$|  mode: ${PKGDOWN_MODE}|"
