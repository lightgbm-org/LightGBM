/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief Metal Shading Language port of the OpenCL histogram256 kernel.
 *        Builds 256-bin gradient/hessian histograms for 4 features at a time.
 *        Double-precision and vendor-specific (NVIDIA/AMD) paths have been removed;
 *        acc_type is always float, acc_int_type is always uint.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------------------
// Compile-time specialization via Metal function constants
// ---------------------------------------------------------------------------
constant int  POWER_FEATURE_WORKGROUPS [[function_constant(0)]];
constant bool CONST_HESSIAN           [[function_constant(1)]];
constant bool ENABLE_ALL_FEATURES     [[function_constant(2)]];
constant bool IGNORE_INDICES          [[function_constant(3)]];

// ---------------------------------------------------------------------------
// Fixed constants (same values as the OpenCL kernel)
// ---------------------------------------------------------------------------
constant constexpr ushort LOCAL_SIZE_0 = 256;
constant constexpr ushort NUM_BINS     = 256;

// acc_type is always float (no FP64 in Metal)
typedef float         acc_type;
typedef uint          acc_int_type;
typedef uint          data_size_t;
typedef float         score_t;

// LOCAL_MEM_SIZE: 4 * (sizeof(uint) + 2 * sizeof(float)) * 256 = 12 288 bytes
constant constexpr uint LOCAL_MEM_SIZE = 4 * (sizeof(uint) + 2 * sizeof(acc_type)) * NUM_BINS;

// ---------------------------------------------------------------------------
// atomic_local_add_f  --  CAS-loop float add into threadgroup memory
// ---------------------------------------------------------------------------
inline void atomic_local_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    float current_f;
    // Unrolled fast path (14 attempts before full loop)
    for (int attempt = 0; attempt < 14; attempt++) {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
    }
    // Full loop
    do {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// within_kernel_reduction256x4
// ---------------------------------------------------------------------------
// Called by histogram256 after the main accumulation.
// We have one sub-histogram in threadgroup memory and need to add the others
// from global memory, then write the merged result to output_buf.
void within_kernel_reduction256x4(
        uchar4 feature_mask,
        device const acc_type* feature4_sub_hist,
        const uint skip_id,
        const uint old_val_f0_cont_bin0,
        const ushort num_sub_hist,
        device acc_type* output_buf,
        threadgroup acc_type* local_hist,
        ushort ltid)
{
    const ushort lsize = LOCAL_SIZE_0;
    // initialize register counters from our threadgroup memory
    acc_type f0_grad_bin = local_hist[ltid * 8];
    acc_type f1_grad_bin = local_hist[ltid * 8 + 1];
    acc_type f2_grad_bin = local_hist[ltid * 8 + 2];
    acc_type f3_grad_bin = local_hist[ltid * 8 + 3];
    acc_type f0_hess_bin = local_hist[ltid * 8 + 4];
    acc_type f1_hess_bin = local_hist[ltid * 8 + 5];
    acc_type f2_hess_bin = local_hist[ltid * 8 + 6];
    acc_type f3_hess_bin = local_hist[ltid * 8 + 7];
    ushort i;

    if (POWER_FEATURE_WORKGROUPS != 0) {
        // add all sub-histograms for 4 features
        device const acc_type* p = feature4_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
            if (feature_mask.w) {
                f0_grad_bin += *p;          p += NUM_BINS;
                f0_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.z) {
                f1_grad_bin += *p;          p += NUM_BINS;
                f1_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.y) {
                f2_grad_bin += *p;          p += NUM_BINS;
                f2_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.x) {
                f3_grad_bin += *p;          p += NUM_BINS;
                f3_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
        }
        // skip the counters we already have
        p += 2 * 4 * NUM_BINS;
        for (i = i + 1; i < num_sub_hist; ++i) {
            if (feature_mask.w) {
                f0_grad_bin += *p;          p += NUM_BINS;
                f0_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.z) {
                f1_grad_bin += *p;          p += NUM_BINS;
                f1_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.y) {
                f2_grad_bin += *p;          p += NUM_BINS;
                f2_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
            if (feature_mask.x) {
                f3_grad_bin += *p;          p += NUM_BINS;
                f3_hess_bin += *p;          p += NUM_BINS;
            }
            else {
                p += 2 * NUM_BINS;
            }
        }
    }
    // now overwrite the threadgroup local_hist for final reduction and output
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // float path: reverse the f3...f0 order to match the real order
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 0] = f3_grad_bin;
    local_hist[0 * 2 * NUM_BINS + ltid * 2 + 1] = f3_hess_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 0] = f2_grad_bin;
    local_hist[1 * 2 * NUM_BINS + ltid * 2 + 1] = f2_hess_bin;
    local_hist[2 * 2 * NUM_BINS + ltid * 2 + 0] = f1_grad_bin;
    local_hist[2 * 2 * NUM_BINS + ltid * 2 + 1] = f1_hess_bin;
    local_hist[3 * 2 * NUM_BINS + ltid * 2 + 0] = f0_grad_bin;
    local_hist[3 * 2 * NUM_BINS + ltid * 2 + 1] = f0_hess_bin;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    i = ltid;
    if (feature_mask.x) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.y) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.z) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
    i += 1 * 2 * NUM_BINS;
    if (feature_mask.w && i < 4 * 2 * NUM_BINS) {
        output_buf[i] = local_hist[i];
        output_buf[i + NUM_BINS] = local_hist[i + NUM_BINS];
    }
}

// ---------------------------------------------------------------------------
// histogram256 kernel
// ---------------------------------------------------------------------------
kernel void histogram256(
        device const uchar4*      feature_data_base   [[buffer(0)]],
        constant const uchar4*    feature_masks       [[buffer(1)]],
        constant const data_size_t& feature_size      [[buffer(2)]],
        device const data_size_t* data_indices        [[buffer(3)]],
        constant const data_size_t& num_data          [[buffer(4)]],
        device const score_t*     ordered_gradients   [[buffer(5)]],
        device const score_t*     ordered_hessians    [[buffer(6)]],
        constant const score_t&   const_hessian_val   [[buffer(7)]],
        device char*              output_buf          [[buffer(8)]],
        device atomic_uint*       sync_counters       [[buffer(9)]],
        device acc_type*          hist_buf_base       [[buffer(10)]],
        uint gtid      [[thread_position_in_grid]],
        uint group_id  [[threadgroup_position_in_grid]],
        uint ltid_u    [[thread_position_in_threadgroup]])
{
    // Cast thread position to ushort to match original kernel types
    const ushort ltid = (ushort)ltid_u;
    const ushort lsize = LOCAL_SIZE_0;

    // Compute global size from group count -- we derive it the same way OpenCL does
    // subglobal_size below is computed from POWER_FEATURE_WORKGROUPS so gsize is not needed.

    // allocate threadgroup memory aligned with float2 for correct alignment
    threadgroup float2 shared_array[LOCAL_MEM_SIZE / sizeof(float2)];

    // ------------------------------------------------------------------
    // clear threadgroup memory
    // ------------------------------------------------------------------
    threadgroup uint* ptr = (threadgroup uint*)shared_array;
    for (uint i = ltid; i < LOCAL_MEM_SIZE / sizeof(uint); i += lsize) {
        ptr[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // gradient/hessian histograms
    // total size: 2 * 4 * 256 * sizeof(float) = 8 KB
    threadgroup acc_type* gh_hist = (threadgroup acc_type*)shared_array;

    // counter histogram (used only when CONST_HESSIAN)
    // total size: 4 * 256 * sizeof(uint) = 4 KB
    threadgroup atomic_uint* cnt_hist = (threadgroup atomic_uint*)(gh_hist + 2 * 4 * NUM_BINS);

    // thread 0,1,2,3 compute histograms for gradients first
    // thread 4,5,6,7 compute histograms for Hessians first, etc.
    uchar is_hessian_first = (ltid >> 2) & 1;

    ushort group_feature = group_id >> POWER_FEATURE_WORKGROUPS;
    // each 2^POWER_FEATURE_WORKGROUPS workgroups process one feature4
    device const uchar4* feature_data = feature_data_base + group_feature * feature_size;
    // size of threads that process this feature4
    const uint subglobal_size = lsize * (1 << POWER_FEATURE_WORKGROUPS);
    // equivalent thread ID in this subgroup for this feature4
    const uint subglobal_tid  = gtid - group_feature * subglobal_size;

    // extract feature mask
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

    // ------------------------------------------------------------------
    // STAGE 1: read feature data, gradient and hessian
    // ------------------------------------------------------------------
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
    feature4_prev = feature4;
    // rotate: as_uchar4(rotate(as_uint(feature4_prev), offset*8))
    {
        uint v = as_type<uint>(feature4_prev);
        uint shift = (uint)offset * 8u;
        if (shift != 0u) {
            v = (v << shift) | (v >> (32u - shift));
        }
        feature4_prev = as_type<uchar4>(v);
    }

    if (!ENABLE_ALL_FEATURES) {
        // rotate feature_mask to match the feature order of each thread
        uint v = as_type<uint>(feature_mask);
        uint shift = (uint)offset * 8u;
        if (shift != 0u) {
            v = (v << shift) | (v >> (32u - shift));
        }
        feature_mask = as_type<uchar4>(v);
    }

    acc_type s3_stat1 = 0.0f, s3_stat2 = 0.0f;
    acc_type s2_stat1 = 0.0f, s2_stat2 = 0.0f;
    acc_type s1_stat1 = 0.0f, s1_stat2 = 0.0f;
    acc_type s0_stat1 = 0.0f, s0_stat2 = 0.0f;

    // ------------------------------------------------------------------
    // Main loop: 2^POWER_FEATURE_WORKGROUPS workgroups process each feature4
    // ------------------------------------------------------------------
    for (uint i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
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
        // rotate feature4
        {
            uint v = as_type<uint>(feature4);
            uint shift = (uint)offset * 8u;
            if (shift != 0u) {
                v = (v << shift) | (v >> (32u - shift));
            }
            feature4 = as_type<uchar4>(v);
        }

        bin = feature4.w;
        if ((bin != feature4_prev.w) && feature_mask.w) {
            bin = feature4_prev.w;
            feature4_prev.w = feature4.w;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s3_stat1);
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

        bin = feature4.z;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.z) && feature_mask.z) {
            bin = feature4_prev.z;
            feature4_prev.z = feature4.z;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s2_stat1);
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
        if (!IGNORE_INDICES) {
            feature4_next = feature_data[ind_next];
        }

        bin = feature4.y;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.y) && feature_mask.y) {
            bin = feature4_prev.y;
            feature4_prev.y = feature4.y;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s1_stat1);
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

        bin = feature4.x;
        offset = (offset + 1) & 0x3;
        if ((bin != feature4_prev.x) && feature_mask.x) {
            bin = feature4_prev.x;
            feature4_prev.x = feature4.x;
            addr = bin * 8 + is_hessian_first * 4 + offset;
            addr2 = addr + 4 - 8 * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s0_stat1);
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
            offset = (ltid & 0x3);
            if (feature_mask.w) {
                bin = feature4.w;
                addr = bin * 4 + offset;
                atomic_fetch_add_explicit(cnt_hist + addr, 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & 0x3;
            if (feature_mask.z) {
                bin = feature4.z;
                addr = bin * 4 + offset;
                atomic_fetch_add_explicit(cnt_hist + addr, 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & 0x3;
            if (feature_mask.y) {
                bin = feature4.y;
                addr = bin * 4 + offset;
                atomic_fetch_add_explicit(cnt_hist + addr, 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & 0x3;
            if (feature_mask.x) {
                bin = feature4.x;
                addr = bin * 4 + offset;
                atomic_fetch_add_explicit(cnt_hist + addr, 1u, memory_order_relaxed);
            }
        }

        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }

    // ------------------------------------------------------------------
    // Flush remaining accumulators
    // ------------------------------------------------------------------
    bin = feature4_prev.w;
    offset = (ltid & 0x3);
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s3_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s3_stat2);
    }

    bin = feature4_prev.z;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s2_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s2_stat2);
    }

    bin = feature4_prev.y;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s1_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s1_stat2);
    }

    bin = feature4_prev.x;
    offset = (offset + 1) & 0x3;
    addr = bin * 8 + is_hessian_first * 4 + offset;
    addr2 = addr + 4 - 8 * is_hessian_first;
    atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr), s0_stat1);
    if (!CONST_HESSIAN) {
        atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), s0_stat2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------
    // Restore feature_mask if not all features enabled
    // ------------------------------------------------------------------
    if (!ENABLE_ALL_FEATURES) {
        feature_mask = feature_masks[group_feature];
    }

    // ------------------------------------------------------------------
    // CONST_HESSIAN final reduction: merge gradient halves and compute hessian from counts
    // ------------------------------------------------------------------
    if (CONST_HESSIAN) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint* cnt_hist_plain = (threadgroup uint*)(gh_hist + 2 * 4 * NUM_BINS);
        offset = ltid & 0x3; // helps avoid LDS bank conflicts
        gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
        gh_hist[ltid * 8 + offset + 4] = const_hessian_val * cnt_hist_plain[ltid * 4 + offset];
        offset = (offset + 1) & 0x3;
        gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
        gh_hist[ltid * 8 + offset + 4] = const_hessian_val * cnt_hist_plain[ltid * 4 + offset];
        offset = (offset + 1) & 0x3;
        gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
        gh_hist[ltid * 8 + offset + 4] = const_hessian_val * cnt_hist_plain[ltid * 4 + offset];
        offset = (offset + 1) & 0x3;
        gh_hist[ltid * 8 + offset] += gh_hist[ltid * 8 + offset + 4];
        gh_hist[ltid * 8 + offset + 4] = const_hessian_val * cnt_hist_plain[ltid * 4 + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------
    // Write output: gradients and hessians histogram for all 4 features
    // ------------------------------------------------------------------
    /* memory layout in gh_hist (total 2 * 4 * 256 * sizeof(float) = 8 KB):
       -----------------------------------------------------------------------------------------------
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0   h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1   h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1
       ...
       g_f0_bin255 g_f1_bin255 g_f2_bin255 g_f3_bin255 h_f0_bin255 h_f1_bin255 h_f2_bin255 h_f3_bin255
       -----------------------------------------------------------------------------------------------
    */
    /* memory layout in cnt_hist (total 4 * 256 * sizeof(uint) = 4 KB):
       -----------------------------------------------
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1
       ...
       c_f0_bin255 c_f1_bin255 c_f2_bin255 c_f3_bin255
       -----------------------------------------------
    */
    // output data in linear order for further reduction
    // output size = 4 (features) * 2 (grad+hess) * 256 (bins) * sizeof(float)
    /* memory layout of output:
       --------------------------------------------
       g_f0_bin0   g_f0_bin1   ...   g_f0_bin255
       h_f0_bin0   h_f0_bin1   ...   h_f0_bin255
       g_f1_bin0   g_f1_bin1   ...   g_f1_bin255
       h_f1_bin0   h_f1_bin1   ...   h_f1_bin255
       g_f2_bin0   g_f2_bin1   ...   g_f2_bin255
       h_f2_bin0   h_f2_bin1   ...   h_f2_bin255
       g_f3_bin0   g_f3_bin1   ...   g_f3_bin255
       h_f3_bin0   h_f3_bin1   ...   h_f3_bin255
       --------------------------------------------
    */
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);

    if (POWER_FEATURE_WORKGROUPS != 0) {
        // Multiple workgroups per feature4: write sub-histogram only.
        // Reduction is done in a separate dispatch (reduce_histogram256).
        device acc_type* output = (device acc_type*)output_buf + group_id * 4 * 2 * NUM_BINS;
        // write gradients and hessians
        device acc_type* ptr_f = output;
        for (ushort j = 0; j < 4; ++j) {
            for (ushort i = ltid; i < 2 * NUM_BINS; i += lsize) {
                // even threads read gradients, odd threads read hessians
                acc_type value = gh_hist[i * 4 + j];
                ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
            }
            ptr_f += 2 * NUM_BINS;
        }
        // Sub-histograms written. Reduction will be done in a separate dispatch.
    } else {
        // only 1 workgroup per feature4, no need for counter -- reduction is a simple copy
        uint old_val = 0; // dummy
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        device acc_type const* feature4_subhists =
                 (device acc_type*)output_buf + output_offset * 4 * 2 * NUM_BINS;
        uint skip_id = group_id ^ output_offset;
        device acc_type* hist_buf = hist_buf_base + feature4_id * 4 * 2 * NUM_BINS;
        within_kernel_reduction256x4(feature_mask, feature4_subhists, skip_id, old_val,
                                     1 << POWER_FEATURE_WORKGROUPS,
                                     hist_buf, (threadgroup acc_type*)shared_array, ltid);
    }
}

// ---------------------------------------------------------------------------
// Reduction kernel: merge sub-histograms from multiple workgroups into final output.
// Each thread handles one float element. Grid size = num_features4 * 4 * 2 * NUM_BINS.
// ---------------------------------------------------------------------------
kernel void reduce_histogram256(
    device const float* sub_histograms [[buffer(0)]],
    device float* output              [[buffer(1)]],
    constant uint& num_sub_hist       [[buffer(2)]],
    constant uint& num_features4      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint elements_per_feature4 = 4 * 2 * NUM_BINS; // = 2048 for 256 bins
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
