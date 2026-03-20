#!/bin/bash

set -e -E -u -o pipefail

RDscriptvalgrind ./.ci/install-r-deps.R --test || exit 1
sh build-cran-package.sh \
  --r-executable=RDvalgrind \
  --no-build-vignettes \
  || exit 1

RDvalgrind CMD INSTALL --preclean --install-tests lightgbm_*.tar.gz || exit 1

cd R-package/tests

ALL_LOGS_FILE="out.log"
VALGRIND_LOGS_FILE="valgrind-logs.log"

RDvalgrind \
  --no-readline \
  --vanilla \
  -d "valgrind --tool=memcheck --leak-check=full --track-origins=yes --gen-suppressions=all" \
  -f testthat.R \
  > ${ALL_LOGS_FILE} 2>&1 || exit 1

cat ${ALL_LOGS_FILE}

echo "writing valgrind output to ${VALGRIND_LOGS_FILE}"
cat ${ALL_LOGS_FILE} | grep -E "^\=" > ${VALGRIND_LOGS_FILE}

bytes_definitely_lost=$(
  cat ${VALGRIND_LOGS_FILE} \
      | grep -E "definitely lost\: .*" \
      | sed 's/^.*definitely lost\: \(.*\) bytes.*$/\1/' \
      | tr -d ","
)
echo "valgrind found ${bytes_definitely_lost} bytes definitely lost"
if [[ ${bytes_definitely_lost} -gt 0 ]]; then
    exit 1
fi

bytes_indirectly_lost=$(
    cat ${VALGRIND_LOGS_FILE} \
    | grep -E "indirectly lost\: .*" \
    | sed 's/^.*indirectly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
echo "valgrind found ${bytes_indirectly_lost} bytes indirectly lost"
if [[ ${bytes_indirectly_lost} -gt 0 ]]; then
    exit 1
fi

# one error caused by a false positive between valgrind and openmp is allowed
# ==2063== 352 bytes in 1 blocks are possibly lost in loss record 153 of 2,709
# ==2063==    at 0x483DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
# ==2063==    by 0x40149CA: allocate_dtv (dl-tls.c:286)
# ==2063==    by 0x40149CA: _dl_allocate_tls (dl-tls.c:532)
# ==2063==    by 0x5702322: allocate_stack (allocatestack.c:622)
# ==2063==    by 0x5702322: pthread_create@@GLIBC_2.2.5 (pthread_create.c:660)
# ==2063==    by 0x56D0DDA: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
# ==2063==    by 0x56C88E0: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
# ==2063==    by 0x1544D29C: LGBM_DatasetCreateFromCSC (c_api.cpp:1286)
# ==2063==    by 0x1546F980: LGBM_DatasetCreateFromCSC_R (lightgbm_R.cpp:91)
# ==2063==    by 0x4941E2F: R_doDotCall (dotcode.c:634)
# ==2063==    by 0x494CCC6: do_dotcall (dotcode.c:1281)
# ==2063==    by 0x499FB01: bcEval (eval.c:7078)
# ==2063==    by 0x498B67F: Rf_eval (eval.c:727)
# ==2063==    by 0x498E414: R_execClosure (eval.c:1895)
bytes_possibly_lost=$(
    cat ${VALGRIND_LOGS_FILE} \
    | grep -E "possibly lost\: .*" \
    | sed 's/^.*possibly lost\: \(.*\) bytes.*$/\1/' \
    | tr -d ","
)
echo "valgrind found ${bytes_possibly_lost} bytes possibly lost"
if [[ ${bytes_possibly_lost} -gt 1104 ]]; then
    exit 1
fi

# ensure 'grep --count' doesn't cause failures
set +e

echo "checking for invalid reads"
invalid_reads=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid read"
)
if [[ ${invalid_reads} -gt 0 ]]; then
    echo "valgrind found invalid reads: ${invalid_reads}"
    exit 1
fi

echo "checking for invalid writes"
invalid_writes=$(
  cat ${VALGRIND_LOGS_FILE} \
    | grep --count -i "Invalid write"
)
if [[ ${invalid_writes} -gt 0 ]]; then
    echo "valgrind found invalid writes: ${invalid_writes}"
    exit 1
fi
