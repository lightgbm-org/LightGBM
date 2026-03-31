/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief Metal Shading Language port of the histogram64 OpenCL kernel.
 *        FP64 path removed; acc_type is always float.
 *        NVIDIA / AMD vendor-specific paths removed.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// -----------------------------------------------------------------------
// Compile-time constants (Metal function constants)
// -----------------------------------------------------------------------
constant int  POWER_FEATURE_WORKGROUPS [[function_constant(0)]];
constant bool CONST_HESSIAN           [[function_constant(1)]];
constant bool ENABLE_ALL_FEATURES     [[function_constant(2)]];
constant bool IGNORE_INDICES          [[function_constant(3)]];

// -----------------------------------------------------------------------
// Fixed constants
// -----------------------------------------------------------------------
#define LOCAL_SIZE_0    256
#define NUM_BINS        64
#define NUM_BANKS       4
#define BANK_BITS       2
#define BANK_MASK       (NUM_BANKS - 1)

// 4 features, each has a gradient and a hessian
#define HG_BIN_MULT     (NUM_BANKS * 4 * 2)     // 32
// 4 features, each has a counter
#define CNT_BIN_MULT    (NUM_BANKS * 4)          // 16

// acc_type is always float (FP64 path removed)
typedef float    acc_type;
typedef uint     acc_int_type;
typedef uint     data_size_t;
typedef float    score_t;

// local memory size in number of uint elements
// 4 * (sizeof(uint) + 2 * sizeof(float)) * NUM_BINS * NUM_BANKS  =  4 * 12 * 64 * 4 = 12288 bytes
// In float2 units: 12288 / 8 = 1536
#define LOCAL_MEM_SIZE  (4 * (sizeof(uint) + 2 * sizeof(acc_type)) * NUM_BINS * NUM_BANKS)

// -----------------------------------------------------------------------
// Helper: rotate a uint left by n bits
// -----------------------------------------------------------------------
inline uint rotate_left(uint v, uint n) {
    n &= 31u;
    if (n == 0u) return v;
    return (v << n) | (v >> (32u - n));
}

// -----------------------------------------------------------------------
// CAS-loop atomic add for float in threadgroup memory
// (vendor-specific Nvidia / AMD paths removed)
// -----------------------------------------------------------------------
inline void atomic_local_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    float current_f;
    // unrolled fast path (14 attempts)
    for (int attempt = 0; attempt < 14; attempt++) {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) return;
    }
    // full loop fallback
    do {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// -----------------------------------------------------------------------
// within_kernel_reduction64x4
// We have one sub-histogram of one feature in registers, and need to
// read other sub-histograms and reduce into the final output.
// -----------------------------------------------------------------------
inline void within_kernel_reduction64x4(
        uchar4 feature_mask,
        device const acc_type* feature4_sub_hist,
        const uint skip_id,
        acc_type g_val, acc_type h_val,
        const ushort num_sub_hist,
        device acc_type* output_buf,
        threadgroup acc_type* local_hist,
        ushort ltid)
{
    const ushort lsize = LOCAL_SIZE_0;
    ushort feature_id = ltid & 3; // range 0 - 3
    const ushort bin_id = ltid >> 2; // range 0 - 63
    ushort i;
    if (POWER_FEATURE_WORKGROUPS != 0) {
        // if there is only 1 work group, no need to do the reduction
        // add all sub-histograms for 4 features
        device const acc_type* p = feature4_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
                g_val += *p;            p += NUM_BINS * 4; // 256 threads working on 4 features' 64 bins
                h_val += *p;            p += NUM_BINS * 4;
        }
        // skip the counters we already have
        p += 2 * 4 * NUM_BINS;
        for (i = i + 1; i < num_sub_hist; ++i) {
                g_val += *p;            p += NUM_BINS * 4;
                h_val += *p;            p += NUM_BINS * 4;
        }
    }
    // now overwrite the local_hist for final reduction and output
    // reverse the f3...f0 order to match the real order
    feature_id = 3 - feature_id;
    local_hist[feature_id * 2 * NUM_BINS + bin_id * 2 + 0] = g_val;
    local_hist[feature_id * 2 * NUM_BINS + bin_id * 2 + 1] = h_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i = ltid;
    if (feature_mask[0] && i < 1 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask[1] && i < 2 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask[2] && i < 3 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask[3] && i < 4 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
    }
}

// -----------------------------------------------------------------------
// Main histogram64 kernel
// -----------------------------------------------------------------------
kernel void histogram64(
        device const uchar4*       feature_data_base  [[buffer(0)]],
        constant uchar4*     feature_masks      [[buffer(1)]],
        constant data_size_t& feature_size       [[buffer(2)]],
        device const data_size_t*  data_indices        [[buffer(3)]],
        constant data_size_t& num_data           [[buffer(4)]],
        device const score_t*      ordered_gradients   [[buffer(5)]],
        device const score_t*      ordered_hessians    [[buffer(6)]],
        constant score_t&    const_hessian       [[buffer(7)]],
        device char*               output_buf          [[buffer(8)]],
        device atomic_uint*        sync_counters       [[buffer(9)]],
        device acc_type*           hist_buf_base       [[buffer(10)]],
        uint gtid       [[thread_position_in_grid]],
        uint gsize      [[threads_per_grid]],
        ushort ltid     [[thread_position_in_threadgroup]],
        uint group_id   [[threadgroup_position_in_grid]])
{
    // ---------------------------------------------------------------
    // Allocate threadgroup memory
    // ---------------------------------------------------------------
    // shared_array aligned as float2 to guarantee correct alignment
    threadgroup float2 shared_array[LOCAL_MEM_SIZE / sizeof(float2)];

    const ushort lsize = LOCAL_SIZE_0;

    // ---------------------------------------------------------------
    // Clear local memory
    // ---------------------------------------------------------------
    threadgroup uint* ptr = (threadgroup uint*)shared_array;
    for (int i = ltid; i < (int)(LOCAL_MEM_SIZE / sizeof(uint)); i += lsize) {
        ptr[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // gradient/hessian histograms
    // each bank: 2 * 4 * 64 * sizeof(float) = 2 KB
    // there are 4 banks (sub-histograms) used by 256 threads total 8 KB
    /* memory layout of gh_hist:
       -----------------------------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0   bk0_g_f2_bin0   bk0_g_f3_bin0   bk0_h_f0_bin0   bk0_h_f1_bin0   bk0_h_f2_bin0   bk0_h_f3_bin0
       bk1_g_f0_bin0   bk1_g_f1_bin0   bk1_g_f2_bin0   bk1_g_f3_bin0   bk1_h_f0_bin0   bk1_h_f1_bin0   bk1_h_f2_bin0   bk1_h_f3_bin0
       bk2_g_f0_bin0   bk2_g_f1_bin0   bk2_g_f2_bin0   bk2_g_f3_bin0   bk2_h_f0_bin0   bk2_h_f1_bin0   bk2_h_f2_bin0   bk2_h_f3_bin0
       bk3_g_f0_bin0   bk3_g_f1_bin0   bk3_g_f2_bin0   bk3_g_f3_bin0   bk3_h_f0_bin0   bk3_h_f1_bin0   bk3_h_f2_bin0   bk3_h_f3_bin0
       bk0_g_f0_bin1   bk0_g_f1_bin1   bk0_g_f2_bin1   bk0_g_f3_bin1   bk0_h_f0_bin1   bk0_h_f1_bin1   bk0_h_f2_bin1   bk0_h_f3_bin1
       ...
       bk3_g_f0_bin63  bk3_g_f1_bin63  bk3_g_f2_bin63  bk3_g_f3_bin63  bk3_h_f0_bin63  bk3_h_f1_bin63  bk3_h_f2_bin63  bk3_h_f3_bin63
       -----------------------------------------------------------------------------------------------
    */
    // with this organization, the LDS/threadgroup memory bank is independent of the bin value
    // all threads within a quarter-wavefront (half-warp/simdgroup) will not have any bank conflict

    threadgroup acc_type* gh_hist = (threadgroup acc_type*)shared_array;
    // counter histogram (only used when CONST_HESSIAN is true)
    // each bank: 4 * 64 * sizeof(uint) = 1 KB
    // there are 4 banks used by 256 threads total 4 KB
    /* memory layout in cnt_hist:
       -----------------------------------------------
       bk0_c_f0_bin0   bk0_c_f1_bin0   bk0_c_f2_bin0   bk0_c_f3_bin0
       bk1_c_f0_bin0   bk1_c_f1_bin0   bk1_c_f2_bin0   bk1_c_f3_bin0
       ...
       bk3_c_f0_bin63  bk3_c_f1_bin63  bk3_c_f2_bin63  bk3_c_f3_bin63
       -----------------------------------------------
    */
    threadgroup uint* cnt_hist = (threadgroup uint*)(gh_hist + 2 * 4 * NUM_BINS * NUM_BANKS);

    // thread 0, 1, 2, 3 compute histograms for gradients first
    // thread 4, 5, 6, 7 compute histograms for Hessians  first
    // etc.
    uchar is_hessian_first = (ltid >> 2) & 1;
    // thread 0-7 write result to bank0, 8-15 to bank1, 16-23 to bank2, 24-31 to bank3
    ushort bank = (ltid >> 3) & BANK_MASK;

    uint group_feature = group_id >> POWER_FEATURE_WORKGROUPS;
    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature (compile-time constant)
    // feature_size is the number of examples per feature
    device const uchar4* feature_data = feature_data_base + group_feature * feature_size;
    // size of threads that process this feature4
    const uint subglobal_size = lsize * (1 << POWER_FEATURE_WORKGROUPS);
    // equivalent thread ID in this subgroup for this feature4
    const uint subglobal_tid  = gtid - group_feature * subglobal_size;
    // extract feature mask, when a byte is set to 0, that feature is disabled
    uchar4 feature_mask;
    if (ENABLE_ALL_FEATURES) {
        feature_mask = uchar4(0xff, 0xff, 0xff, 0xff);
    } else {
        feature_mask = feature_masks[group_feature];
    }
    // exit if all features are masked
    if (!as_type<uint>(feature_mask)) {
        return;
    }

    // STAGE 1: read feature data, and gradient and hessian
    // 4 features stored in a tuple MSB...(0, 1, 2, 3)...LSB
    uchar4 feature4;
    uchar4 feature4_next;
    uchar4 feature4_prev;
    // offset used to rotate feature4 vector
    ushort offset = (ltid & 0x3);
    // store gradient and hessian
    float stat1, stat2;
    float stat1_next, stat2_next;
    ushort bin, addr, addr2;
    data_size_t ind;
    data_size_t ind_next;
    stat1 = ordered_gradients[subglobal_tid];
    if (!CONST_HESSIAN) {
        stat2 = ordered_hessians[subglobal_tid];
    }
    if (IGNORE_INDICES) {
        ind = subglobal_tid;
    } else {
        ind = data_indices[subglobal_tid];
    }
    feature4 = feature_data[ind];
    feature4 = as_type<uchar4>(as_type<uint>(feature4) & 0x3f3f3f3f);
    feature4_prev = feature4;
    feature4_prev = as_type<uchar4>(rotate_left(as_type<uint>(feature4_prev), (uint)offset * 8u));
    if (!ENABLE_ALL_FEATURES) {
        // rotate feature_mask to match the feature order of each thread
        feature_mask = as_type<uchar4>(rotate_left(as_type<uint>(feature_mask), (uint)offset * 8u));
    }
    acc_type s3_stat1 = 0.0f, s3_stat2 = 0.0f;
    acc_type s2_stat1 = 0.0f, s2_stat2 = 0.0f;
    acc_type s1_stat1 = 0.0f, s1_stat2 = 0.0f;
    acc_type s0_stat1 = 0.0f, s0_stat2 = 0.0f;

    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (uint i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need boundary check because we have made the buffer larger
        stat1_next = ordered_gradients[i + subglobal_size];
        if (!CONST_HESSIAN) {
            stat2_next = ordered_hessians[i + subglobal_size];
        }
        if (IGNORE_INDICES) {
            // we need to check bounds here
            ind_next = i + subglobal_size < num_data ? i + subglobal_size : i;
            // start load next feature as early as possible
            feature4_next = feature_data[ind_next];
        } else {
            ind_next = data_indices[i + subglobal_size];
        }
        if (!CONST_HESSIAN) {
            // swap gradient and hessian for threads 4, 5, 6, 7
            float tmp = stat1;
            stat1 = is_hessian_first ? stat2 : stat1;
            stat2 = is_hessian_first ? tmp   : stat2;
        }

        // STAGE 2: accumulate gradient and hessian
        offset = (ltid & 0x3);
        feature4 = as_type<uchar4>(rotate_left(as_type<uint>(feature4), (uint)offset * 8u));
        bin = feature4[3];
        if ((bin != feature4_prev[3]) && feature_mask[3]) {
            bin = feature4_prev[3];
            feature4_prev[3] = feature4[3];
            addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's Hessians  for example 4, 5, 6, 7
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s3_stat1);
            // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's gradients for example 4, 5, 6, 7
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s3_stat2);
            }
            s3_stat1 = stat1;
            s3_stat2 = stat2;
        }
        else {
            s3_stat1 += stat1;
            s3_stat2 += stat2;
        }

        bin = feature4[2];
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev[2]) && feature_mask[2]) {
            bin = feature4_prev[2];
            feature4_prev[2] = feature4[2];
            addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's Hessians  for example 4, 5, 6, 7
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s2_stat1);
            // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's gradients for example 4, 5, 6, 7
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s2_stat2);
            }
            s2_stat1 = stat1;
            s2_stat2 = stat2;
        }
        else {
            s2_stat1 += stat1;
            s2_stat2 += stat2;
        }

        // prefetch the next iteration variables
        // we don't need boundary check because if it is out of boundary, ind_next = 0
        if (!IGNORE_INDICES) {
            feature4_next = feature_data[ind_next];
        }

        bin = feature4[1] & 0x3f;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev[1]) && feature_mask[1]) {
            bin = feature4_prev[1];
            feature4_prev[1] = feature4[1];
            addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's Hessians  for example 4, 5, 6, 7
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s1_stat1);
            // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's gradients for example 4, 5, 6, 7
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s1_stat2);
            }
            s1_stat1 = stat1;
            s1_stat2 = stat2;
        }
        else {
            s1_stat1 += stat1;
            s1_stat2 += stat2;
        }

        bin = feature4[0];
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev[0]) && feature_mask[0]) {
            bin = feature4_prev[0];
            feature4_prev[0] = feature4[0];
            addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's gradients for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's Hessians  for example 4, 5, 6, 7
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s0_stat1);
            // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's Hessians  for example 0, 1, 2, 3
            // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's gradients for example 4, 5, 6, 7
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s0_stat2);
            }
            s0_stat1 = stat1;
            s0_stat2 = stat2;
        }
        else {
            s0_stat1 += stat1;
            s0_stat2 += stat2;
        }
        if (CONST_HESSIAN) {
            // STAGE 3: accumulate counter
            // there are 4 counters for 4 features
            // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's counts for example 0, 1, 2, 3
            offset = (ltid & 0x3);
            if (feature_mask[3]) {
                bin = feature4[3];
                addr = bin * CNT_BIN_MULT + bank * 4 + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr), 1u, memory_order_relaxed);
            }
            // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's counts for example 0, 1, 2, 3
            offset = (offset + 1) & 0x3;
            if (feature_mask[2]) {
                bin = feature4[2];
                addr = bin * CNT_BIN_MULT + bank * 4 + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr), 1u, memory_order_relaxed);
            }
            // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's counts for example 0, 1, 2, 3
            offset = (offset + 1) & 0x3;
            if (feature_mask[1]) {
                bin = feature4[1];
                addr = bin * CNT_BIN_MULT + bank * 4 + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr), 1u, memory_order_relaxed);
            }
            // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's counts for example 0, 1, 2, 3
            offset = (offset + 1) & 0x3;
            if (feature_mask[0]) {
                bin = feature4[0];
                addr = bin * CNT_BIN_MULT + bank * 4 + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr), 1u, memory_order_relaxed);
            }
        }
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
        feature4 = as_type<uchar4>(as_type<uint>(feature4) & 0x3f3f3f3f);
    }

    // ---------------------------------------------------------------
    // Flush remaining accumulators
    // ---------------------------------------------------------------
    bin = feature4_prev[3];
    offset = (ltid & 0x3);
    addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s3_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s3_stat2);
    }

    bin = feature4_prev[2];
    offset = (offset + 1) & 0x3;
    addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s2_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s2_stat2);
    }

    bin = feature4_prev[1];
    offset = (offset + 1) & 0x3;
    addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s1_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s1_stat2);
    }

    bin = feature4_prev[0];
    offset = (offset + 1) & 0x3;
    addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s0_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s0_stat2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------------------------------------------------------
    // Restore feature_mask if not all features enabled
    // ---------------------------------------------------------------
    if (!ENABLE_ALL_FEATURES) {
        feature_mask = feature_masks[group_feature];
    }

    // ---------------------------------------------------------------
    // Reduce the 4 banks of sub-histograms into 1
    // ---------------------------------------------------------------
    /* memory layout of gh_hist:
       -----------------------------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0   bk0_g_f2_bin0   bk0_g_f3_bin0   bk0_h_f0_bin0   bk0_h_f1_bin0   bk0_h_f2_bin0   bk0_h_f3_bin0
       bk1_g_f0_bin0   bk1_g_f1_bin0   bk1_g_f2_bin0   bk1_g_f3_bin0   bk1_h_f0_bin0   bk1_h_f1_bin0   bk1_h_f2_bin0   bk1_h_f3_bin0
       bk2_g_f0_bin0   bk2_g_f1_bin0   bk2_g_f2_bin0   bk2_g_f3_bin0   bk2_h_f0_bin0   bk2_h_f1_bin0   bk2_h_f2_bin0   bk2_h_f3_bin0
       bk3_g_f0_bin0   bk3_g_f1_bin0   bk3_g_f2_bin0   bk3_g_f3_bin0   bk3_h_f0_bin0   bk3_h_f1_bin0   bk3_h_f2_bin0   bk3_h_f3_bin0
       ...
       bk3_g_f0_bin63  bk3_g_f1_bin63  bk3_g_f2_bin63  bk3_g_f3_bin63  bk3_h_f0_bin63  bk3_h_f1_bin63  bk3_h_f2_bin63  bk3_h_f3_bin63
       -----------------------------------------------------------------------------------------------
    */
    /* memory layout in cnt_hist:
       -----------------------------------------------
       bk0_c_f0_bin0   bk0_c_f1_bin0   bk0_c_f2_bin0   bk0_c_f3_bin0
       bk1_c_f0_bin0   bk1_c_f1_bin0   bk1_c_f2_bin0   bk1_c_f3_bin0
       ...
       bk3_c_f0_bin63  bk3_c_f1_bin63  bk3_c_f2_bin63  bk3_c_f3_bin63
       -----------------------------------------------
    */
    acc_type g_val = 0.0f;
    acc_type h_val = 0.0f;
    uint cnt_val = 0;
    // 256 threads, working on 4 features and 64 bins,
    // so each thread has an independent feature/bin to work on.
    const ushort feature_id = ltid & 3; // range 0 - 3
    const ushort bin_id = ltid >> 2; // range 0 - 63
    offset = (ltid >> 2) & BANK_MASK; // helps avoid LDS bank conflicts
    for (int i = 0; i < NUM_BANKS; ++i) {
        ushort bank_id = (i + offset) & BANK_MASK;
        g_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 8 + feature_id];
        h_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 8 + feature_id + 4];
        if (CONST_HESSIAN) {
            cnt_val += cnt_hist[bin_id * CNT_BIN_MULT + bank_id * 4 + feature_id];
        }
    }
    // now thread 0 - 3 holds feature 0, 1, 2, 3's gradient, hessian and count bin 0
    // now thread 4 - 7 holds feature 0, 1, 2, 3's gradient, hessian and count bin 1
    // etc.

    if (CONST_HESSIAN) {
        g_val += h_val;
        h_val = cnt_val * const_hessian;
    }

    // write to output
    // write gradients and Hessians histogram for all 4 features
    // output data in linear order for further reduction
    // output size = 4 (features) * 3 (counters) * 64 (bins) * sizeof(float)
    /* memory layout of output:
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1
       ...
       g_f0_bin63  g_f1_bin63  g_f2_bin63  g_f3_bin63
       h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0
       h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1
       ...
       h_f0_bin63  h_f1_bin63  h_f2_bin63  h_f3_bin63
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1
       ...
       c_f0_bin63  c_f1_bin63  c_f2_bin63  c_f3_bin63
    */
    // if there is only one workgroup processing this feature4, don't even need to write
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);
    if (POWER_FEATURE_WORKGROUPS != 0) {
        // Multiple workgroups per feature4: write sub-histogram only.
        // Reduction is done in a separate dispatch (reduce_histogram64).
        device acc_type* output = (device acc_type*)output_buf + group_id * 4 * 2 * NUM_BINS;
        // write gradients for 4 features
        output[0 * 4 * NUM_BINS + ltid] = g_val;
        // write Hessians for 4 features
        output[1 * 4 * NUM_BINS + ltid] = h_val;
        // Sub-histograms written. Reduction will be done in a separate dispatch.
    } else {
        // only 1 work group, no need to increase counter
        // the reduction will become a simple copy
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        device acc_type const* feature4_subhists =
                 (device acc_type*)output_buf + output_offset * 4 * 2 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id = group_id ^ output_offset;
        // locate output histogram location for this feature4
        device acc_type* hist_buf = hist_buf_base + feature4_id * 4 * 2 * NUM_BINS;
        within_kernel_reduction64x4(feature_mask, feature4_subhists, skip_id, g_val, h_val,
                                    1 << POWER_FEATURE_WORKGROUPS, hist_buf, (threadgroup acc_type*)shared_array, ltid);
    }
}

// -----------------------------------------------------------------------
// Reduction kernel: merge sub-histograms from multiple workgroups into final output.
// Each thread handles one float element. Grid size = num_features4 * 4 * 2 * NUM_BINS.
// -----------------------------------------------------------------------
kernel void reduce_histogram64(
    device const float* sub_histograms [[buffer(0)]],
    device float* output              [[buffer(1)]],
    constant uint& num_sub_hist       [[buffer(2)]],
    constant uint& num_features4      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint elements_per_feature4 = 4 * 2 * NUM_BINS; // = 512 for 64 bins
    uint feature4_id = tid / elements_per_feature4;
    uint element_id = tid % elements_per_feature4;

    if (feature4_id >= num_features4) return;

    float sum = 0.0f;
    for (uint s = 0; s < num_sub_hist; ++s) {
        // Sub-histogram layout: sub_histograms[(feature4_id * num_sub_hist + s) * elements_per_feature4 + element_id]
        sum += sub_histograms[(feature4_id * num_sub_hist + s) * elements_per_feature4 + element_id];
    }

    output[feature4_id * elements_per_feature4 + element_id] = sum;
}
