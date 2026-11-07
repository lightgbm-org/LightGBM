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
/*!
 * The NCCL declarations in this file are derived from NVIDIA NCCL
 * (https://github.com/NVIDIA/nccl), distributed under the BSD 3-Clause License:
 *
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION, Lawrence Berkeley National
 *   Laboratory, the U.S. Department of Energy, nor the names of their
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */
/*!
 * Minimal stub of the NCCL API used by LightGBM. NCCL has no official Windows
 * distribution, but LightGBM only calls NCCL when num_gpu > 1 (see
 * Boosting::CreateBoosting). The stub therefore only satisfies the compiler
 * and linker for single-GPU builds; its implementation aborts if any NCCL
 * operation is called.
 *
 * ncclUniqueId must remain NCCL_UNIQUE_ID_BYTES bytes because LightGBM uses
 * sizeof(ncclUniqueId) when exchanging the ID with Network::Allgather.
 */
#ifndef LIGHTGBM_CMAKE_NCCL_STUB_NCCL_H_
#define LIGHTGBM_CMAKE_NCCL_STUB_NCCL_H_

#include <cuda_runtime.h>
#include <stddef.h>

#define NCCL_UNIQUE_ID_BYTES 128

typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
typedef struct ncclComm* ncclComm_t;

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

typedef enum {
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
  ncclBfloat16 = 9
} ncclDataType_t;

typedef enum {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4
} ncclRedOp_t;

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream);
const char* ncclGetErrorString(ncclResult_t result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIGHTGBM_CMAKE_NCCL_STUB_NCCL_H_
