/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nccl.h>

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void NCCLStubFail(const char* function) {
  std::fprintf(stderr,
               "[LightGBM] FATAL: %s requires NCCL, but this build uses the "
               "single-GPU NCCL stub. Install NCCL and rebuild LightGBM for "
               "multi-GPU training.\n", function);
  std::abort();
}

}  // namespace

ncclResult_t ncclGroupStart(void) {
  NCCLStubFail("ncclGroupStart");
}

ncclResult_t ncclGroupEnd(void) {
  NCCLStubFail("ncclGroupEnd");
}

ncclResult_t ncclGetUniqueId(ncclUniqueId*) {
  NCCLStubFail("ncclGetUniqueId");
}

ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) {
  NCCLStubFail("ncclCommInitRank");
}

ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t,
                           ncclComm_t, cudaStream_t) {
  NCCLStubFail("ncclAllReduce");
}

const char* ncclGetErrorString(ncclResult_t) {
  return "NCCL is unavailable in this single-GPU build";
}
