#!/bin/bash
#
# [description]
#     Helper script for checking versions in the dynamic symbol table.
#     This script checks that LightGBM library is linked to the appropriate symbol versions.
#     Linking to newer symbol versions at compile time is problematic because it could result
#     in built artifacts being unusable on older platforms.
#
#     Version history for these symbols can be found at the following:
#         * GLIBC: https://sourceware.org/glibc/wiki/Glibc%20Timeline
#         * GLIBCXX: https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
#         * OMP/GOMP: https://github.com/gcc-mirror/gcc/blob/master/libgomp/libgomp.map
#
# [usage]
#     check-dynamic-dependencies.sh <PATH>
#
# PATH: Path to the file.
#       Path to the file with the dynamic symbol table entries of the file
#       (result of `objdump -T` command).

set -e -E -u -o pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <PATH>"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

awk '
BEGIN {
    glibc_count = 0
    glibcxx_count = 0
    gomp_count = 0

    has_error = 0
}

# --- Check GLIBC ---
/0{16}[ \t\(]+GLIBC_[0-9]+\.[0-9]+/ {
    match($0, /GLIBC_([0-9]+)\.([0-9]+)/, parts)
    if (RSTART > 0) {
        match($0, /GLIBC_[0-9]+\.[0-9]+/)
        ver_str = substr($0, RSTART+6, RLENGTH-6)  # skip "GLIBC_"
        split(ver_str, v, ".")
        major = v[1] + 0
        minor = v[2] + 0

        glibc_count++

        if (major > 2 || (major == 2 && minor > 28)) {
            print "Error: found unexpected GLIBC version: \x27" major "." minor "\x27"
            has_error = 1
        }
    }
}

# --- Check GLIBCXX ---
/0{16}[ \t\(]+GLIBCXX_[0-9]+\.[0-9]+/ {
    match($0, /GLIBCXX_[0-9]+\.[0-9]+(\.[0-9]+)?/)
    if (RSTART > 0) {
        ver_str = substr($0, RSTART+8, RLENGTH-8)  # skip "GLIBCXX_"
        n = split(ver_str, v, ".")
        major = v[1] + 0
        minor = v[2] + 0
        patch = (n >= 3) ? v[3] + 0 : 0
        patch_str = (n >= 3) ? v[3] : ""

        glibcxx_count++

        msg_ver = major "." minor
        if (n >= 3) msg_ver = msg_ver "." patch

        if (major != 3 || minor != 4) {
             print "Error: found unexpected GLIBCXX version: \x27" msg_ver "\x27"
             has_error = 1
        }
        if (n >= 3 && patch > 22) {
             print "Error: found unexpected GLIBCXX version: \x27" msg_ver "\x27"
             has_error = 1
        }
    }
}

# --- Check OMP/GOMP ---
/0{16}[ \t\(]+G?OMP_[0-9]+\.[0-9]+/ {
    match($0, /G?OMP_[0-9]+\.[0-9]+/)
    if (RSTART > 0) {
        full_match = substr($0, RSTART, RLENGTH)
        us_idx = index(full_match, "_")
        ver_str = substr(full_match, us_idx + 1)

        split(ver_str, v, ".")
        major = v[1] + 0
        minor = v[2] + 0

        gomp_count++

        if (major > 4 || (major == 4 && minor > 5)) {
            print "Error: found unexpected OMP/GOMP version: \x27" major "." minor "\x27"
            has_error = 1
        }
    }
}

END {
    if (glibc_count <= 1) {
        print "Error: Not enough GLIBC symbols found (found " glibc_count ", expected > 1)"
        has_error = 1
    }
    if (glibcxx_count <= 1) {
        print "Error: Not enough GLIBCXX symbols found (found " glibcxx_count ", expected > 1)"
        has_error = 1
    }
    if (gomp_count <= 1) {
        print "Error: Not enough OMP/GOMP symbols found (found " gomp_count ", expected > 1)"
        has_error = 1
    }

    if (has_error == 1) {
        exit 1
    }
}
' "$INPUT_FILE"
