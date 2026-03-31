/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief Metal Shading Language port of the OpenCL histogram16 kernel.
 *        FP64 path removed; acc_type is always float, acc_int_type is always uint.
 *        NVIDIA / AMD vendor-specific paths removed.
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
// Fixed constants (same semantics as the OpenCL kernel)
// ---------------------------------------------------------------------------
constant constexpr ushort LOCAL_SIZE_0         = 256;
constant constexpr ushort NUM_BINS             = 16;
constant constexpr ushort DWORD_FEATURES       = 8;
constant constexpr ushort FEATURE_BITS         = 4;       // sizeof(uchar4)*8 / DWORD_FEATURES
constant constexpr ushort DWORD_FEATURES_MASK  = DWORD_FEATURES - 1;  // 7
constant constexpr ushort LOG2_DWORD_FEATURES  = 3;
constant constexpr ushort NUM_BANKS            = 8;
constant constexpr ushort BANK_BITS            = 3;
constant constexpr ushort BANK_MASK            = NUM_BANKS - 1;       // 7
// 8 features, each has a gradient and a hessian
constant constexpr ushort HG_BIN_MULT          = NUM_BANKS * DWORD_FEATURES * 2;  // 128
// 8 features, each has a counter
constant constexpr ushort CNT_BIN_MULT         = NUM_BANKS * DWORD_FEATURES;      // 64
// local memory size in bytes:
//   DWORD_FEATURES * (sizeof(uint) + 2 * sizeof(float)) * NUM_BINS * NUM_BANKS
//   = 8 * (4 + 8) * 16 * 8 = 12288 bytes
constant constexpr uint   LOCAL_MEM_SIZE       = DWORD_FEATURES * (sizeof(uint) + 2 * sizeof(float)) * NUM_BINS * NUM_BANKS;

// Typedefs matching the OpenCL kernel
typedef uint  data_size_t;
typedef float score_t;
typedef float acc_type;
typedef uint  acc_int_type;

// ---------------------------------------------------------------------------
// uchar8 replacement -- Metal lacks a native uchar8 vector type.
// This packed struct has the same 8-byte layout as OpenCL uchar8 and can be
// read directly from a device buffer that was written by the host as uchar[8].
// ---------------------------------------------------------------------------
struct uchar8_t {
    uchar s[8];

    thread uchar& operator[](int i) thread { return s[i]; }
    uchar operator[](int i) const thread { return s[i]; }
};

static inline uchar8_t make_uchar8(uchar v0, uchar v1, uchar v2, uchar v3,
                                    uchar v4, uchar v5, uchar v6, uchar v7) {
    uchar8_t r;
    r.s[0] = v0; r.s[1] = v1; r.s[2] = v2; r.s[3] = v3;
    r.s[4] = v4; r.s[5] = v5; r.s[6] = v6; r.s[7] = v7;
    return r;
}

// Check if all 8 bytes are zero (equivalent to OpenCL !as_ulong(feature_mask))
static inline bool uchar8_all_zero(uchar8_t m) {
    uint lo = as_type<uint>(uchar4(m.s[0], m.s[1], m.s[2], m.s[3]));
    uint hi = as_type<uint>(uchar4(m.s[4], m.s[5], m.s[6], m.s[7]));
    return (lo | hi) == 0u;
}

// ---------------------------------------------------------------------------
// atomic_local_add_f  --  CAS-loop atomic float add on threadgroup memory
// ---------------------------------------------------------------------------
inline void atomic_local_add_f(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    float current_f;
    // Unrolled fast-path: 14 attempts before falling through to the full loop
    for (int attempt = 0; attempt < 14; attempt++) {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed)) return;
    }
    do {
        current_f = as_type<float>(expected);
        next = as_type<uint>(current_f + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// within_kernel_reduction16x8
// ---------------------------------------------------------------------------
// Called by the histogram16 kernel.
// We have one sub-histogram of one feature in registers and need to read
// others, then write the final result.
void within_kernel_reduction16x8(
        uchar8_t feature_mask,
        device const acc_type* feature4_sub_hist,
        const uint skip_id,
        acc_type stat_val,
        const ushort num_sub_hist,
        device acc_type* output_buf,
        threadgroup acc_type* local_hist,
        ushort ltid)
{
    const ushort lsize = LOCAL_SIZE_0;
    ushort feature_id = ltid & DWORD_FEATURES_MASK; // range 0 - 7
    uchar is_hessian_first = (ltid >> LOG2_DWORD_FEATURES) & 1; // hessian or gradient
    ushort bin_id = ltid >> (LOG2_DWORD_FEATURES + 1); // range 0 - 16
    ushort i;

    if (POWER_FEATURE_WORKGROUPS != 0) {
        // if there is only 1 work group, no need to do the reduction
        // add all sub-histograms for 8 features
        device const acc_type* p = feature4_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
            // 256 threads working on 8 features' 16 bins, gradient and Hessian
            stat_val += *p;
            p += NUM_BINS * DWORD_FEATURES * 2;
        }
        // skip the counters we already have
        p += 2 * DWORD_FEATURES * NUM_BINS;
        for (i = i + 1; i < num_sub_hist; ++i) {
            stat_val += *p;
            p += NUM_BINS * DWORD_FEATURES * 2;
        }
    }

    // now overwrite the local_hist for final reduction and output
    // reverse the f7...f0 order to match the real order
    feature_id = DWORD_FEATURES_MASK - feature_id;
    local_hist[feature_id * 2 * NUM_BINS + bin_id * 2 + is_hessian_first] = stat_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (i = ltid; i < DWORD_FEATURES * 2 * NUM_BINS; i += lsize) {
        output_buf[i] = local_hist[i];
    }
}

// ---------------------------------------------------------------------------
// histogram16 kernel
// ---------------------------------------------------------------------------
kernel void histogram16(
        device const uchar4*       feature_data_base   [[buffer(0)]],
        device const uchar8_t*     feature_masks       [[buffer(1)]],
        constant data_size_t&      feature_size         [[buffer(2)]],
        device const data_size_t*  data_indices         [[buffer(3)]],
        constant data_size_t&      num_data             [[buffer(4)]],
        device const score_t*      ordered_gradients    [[buffer(5)]],
        device const score_t*      ordered_hessians     [[buffer(6)]],
        constant score_t&          const_hessian_val    [[buffer(7)]],
        device char*               output_buf           [[buffer(8)]],
        device atomic_uint*        sync_counters        [[buffer(9)]],
        device acc_type*           hist_buf_base        [[buffer(10)]],
        uint gtid      [[thread_position_in_grid]],
        uint gsize     [[threads_per_grid]],
        uint ltid_uint [[thread_position_in_threadgroup]],
        uint group_id  [[threadgroup_position_in_grid]])
{
    // Cast thread-position values to the ushort types used throughout
    const ushort ltid  = (ushort)ltid_uint;
    const ushort lsize = LOCAL_SIZE_0;

    // -----------------------------------------------------------------------
    // Allocate threadgroup (local) memory
    // -----------------------------------------------------------------------
    // Aligned with float2 to guarantee correct alignment
    threadgroup float2 shared_array[LOCAL_MEM_SIZE / sizeof(float2)];

    // -----------------------------------------------------------------------
    // Clear local memory
    // -----------------------------------------------------------------------
    threadgroup uint* ptr = (threadgroup uint*)shared_array;
    for (int i = ltid; i < (int)(LOCAL_MEM_SIZE / sizeof(uint)); i += lsize) {
        ptr[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Gradient / hessian histograms
    // -----------------------------------------------------------------------
    /* memory layout of gh_hist:
       -------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0  ...  bk0_g_f7_bin0
       bk0_h_f0_bin0   bk0_h_f1_bin0  ...  bk0_h_f7_bin0
       bk1_g_f0_bin0   ...
       ...
       bk7_h_f0_bin15  ...                   bk7_h_f7_bin15
       -------------------------------------------------------------------------
    */
    // with this organization, the LDS bank is independent of the bin value;
    // all threads within a quarter-wavefront (half-warp) have no bank conflict.
    threadgroup acc_type* gh_hist = (threadgroup acc_type*)shared_array;

    // counter histogram (only used when CONST_HESSIAN is true)
    /* memory layout of cnt_hist:
       bk0_c_f0_bin0  bk0_c_f1_bin0  ...  bk0_c_f7_bin0
       bk1_c_f0_bin0  ...
       ...
       bk7_c_f0_bin15 ...                  bk7_c_f7_bin15
    */
    threadgroup uint* cnt_hist = (threadgroup uint*)(gh_hist + 2 * DWORD_FEATURES * NUM_BINS * NUM_BANKS);

    // -----------------------------------------------------------------------
    // Per-thread state
    // -----------------------------------------------------------------------
    // thread 0-7 compute gradients first; thread 8-15 compute Hessians first; etc.
    uchar is_hessian_first = (ltid >> LOG2_DWORD_FEATURES) & 1;
    // thread 0-15 -> bank0, 16-31 -> bank1, ... 240-255 -> bank7 (wrapping)
    ushort bank = (ltid >> (LOG2_DWORD_FEATURES + 1)) & BANK_MASK;

    ushort group_feature = group_id >> POWER_FEATURE_WORKGROUPS;
    // each 2^POWER_FEATURE_WORKGROUPS workgroups process one feature tuple
    device const uchar4* feature_data = feature_data_base + group_feature * feature_size;
    // size of threads that process this feature4
    const uint subglobal_size = lsize * (1 << POWER_FEATURE_WORKGROUPS);
    // equivalent thread ID in this subgroup for this feature4
    const uint subglobal_tid  = gtid - group_feature * subglobal_size;

    // extract feature mask; when a byte is 0 that feature is disabled
    uchar8_t feature_mask;
    if (ENABLE_ALL_FEATURES) {
        feature_mask = make_uchar8(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
    } else {
        feature_mask = feature_masks[group_feature];
    }

    // exit if all features are masked
    if (uchar8_all_zero(feature_mask)) {
        return;
    }

    // -----------------------------------------------------------------------
    // STAGE 1: read feature data, gradient and hessian
    // -----------------------------------------------------------------------
    uchar4 feature4;
    uchar4 feature4_next;
    // offset used to rotate feature4 vector
    ushort offset = (ltid & DWORD_FEATURES_MASK);

    if (!ENABLE_ALL_FEATURES) {
        // rotate feature_mask to match the feature order of each thread
        // OpenCL: feature_mask = as_uchar8(rotate(as_ulong(feature_mask), (ulong)offset*8));
        // rotate(ulong, n) is a LEFT rotate on the 64-bit value by n bits.
        // On little-endian, uchar8 element [0] is the LSB of the ulong.
        // Left-rotating by offset*8 bits means: new_element[r] = old_element[(r - offset) & 7]
        uchar temp[8];
        for (int k = 0; k < 8; k++) temp[k] = feature_mask.s[k];
        for (int r = 0; r < 8; r++) {
            feature_mask.s[r] = temp[(r - offset) & 7];
        }
    }

    // store gradient and hessian
    float stat1, stat2;
    float stat1_next, stat2_next;
    ushort bin, addr_val, addr2;
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

    // -----------------------------------------------------------------------
    // Main loop: there are 2^POWER_FEATURE_WORKGROUPS workgroups per feature4
    // -----------------------------------------------------------------------
    for (uint i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // buffer is made larger so no boundary check needed for gradient/hessian
        stat1_next = ordered_gradients[i + subglobal_size];
        if (!CONST_HESSIAN) {
            stat2_next = ordered_hessians[i + subglobal_size];
        }
        if (IGNORE_INDICES) {
            // we need to check bounds here
            ind_next = (i + subglobal_size < num_data) ? (i + subglobal_size) : i;
            // start load next feature as early as possible
            feature4_next = feature_data[ind_next];
        } else {
            ind_next = data_indices[i + subglobal_size];
        }

        if (!CONST_HESSIAN) {
            // swap gradient and hessian for threads 8-15, 24-31, etc.
            float tmp = stat1;
            stat1 = is_hessian_first ? stat2 : stat1;
            stat2 = is_hessian_first ? tmp   : stat2;
        }

        // -------------------------------------------------------------------
        // STAGE 2: accumulate gradient and hessian
        // -------------------------------------------------------------------
        offset = (ltid & DWORD_FEATURES_MASK);
        // Manual rotate: rotate as_uint(feature4) left by (offset * FEATURE_BITS) bits
        {
            uint v = as_type<uint>(feature4);
            uint shift = (uint)(offset * FEATURE_BITS) & 31u;
            if (shift != 0u) {
                v = (v << shift) | (v >> (32u - shift));
            }
            feature4 = as_type<uchar4>(v);
        }

        // Feature 7 (OpenCL .s7)
        if (feature_mask.s[7]) {
            bin = feature4[3] >> 4;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 6 (OpenCL .s6)
        if (feature_mask.s[6]) {
            bin = feature4[3] & 0xf;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 5 (OpenCL .s5)
        if (feature_mask.s[5]) {
            bin = feature4[2] >> 4;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 4 (OpenCL .s4)
        if (feature_mask.s[4]) {
            bin = feature4[2] & 0xf;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }

        // prefetch the next iteration variables (non-IGNORE_INDICES path)
        // buffer is made larger so if out of boundary, ind_next = 0
        if (!IGNORE_INDICES) {
            feature4_next = feature_data[ind_next];
        }

        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 3 (OpenCL .s3)
        if (feature_mask.s[3]) {
            bin = feature4[1] >> 4;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 2 (OpenCL .s2)
        if (feature_mask.s[2]) {
            bin = feature4[1] & 0xf;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 1 (OpenCL .s1)
        if (feature_mask.s[1]) {
            bin = feature4[0] >> 4;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }
        offset = (offset + 1) & DWORD_FEATURES_MASK;

        // Feature 0 (OpenCL .s0)
        if (feature_mask.s[0]) {
            bin = feature4[0] & 0xf;
            addr_val = bin * HG_BIN_MULT + bank * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + offset;
            addr2 = addr_val + DWORD_FEATURES - 2 * DWORD_FEATURES * is_hessian_first;
            atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr_val), stat1);
            if (!CONST_HESSIAN) {
                atomic_local_add_f((threadgroup atomic_uint*)(gh_hist + addr2), stat2);
            }
        }

        // -------------------------------------------------------------------
        // STAGE 3: accumulate counter (CONST_HESSIAN path only)
        // -------------------------------------------------------------------
        if (CONST_HESSIAN) {
            offset = (ltid & DWORD_FEATURES_MASK);
            if (feature_mask.s[7]) {
                bin = feature4[3] >> 4;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[6]) {
                bin = feature4[3] & 0xf;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[5]) {
                bin = feature4[2] >> 4;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[4]) {
                bin = feature4[2] & 0xf;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[3]) {
                bin = feature4[1] >> 4;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[2]) {
                bin = feature4[1] & 0xf;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[1]) {
                bin = feature4[0] >> 4;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
            offset = (offset + 1) & DWORD_FEATURES_MASK;
            if (feature_mask.s[0]) {
                bin = feature4[0] & 0xf;
                addr_val = bin * CNT_BIN_MULT + bank * DWORD_FEATURES + offset;
                atomic_fetch_add_explicit((threadgroup atomic_uint*)(cnt_hist + addr_val), 1u, memory_order_relaxed);
            }
        }

        // advance to next iteration
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Restore feature_mask if it was rotated
    // -----------------------------------------------------------------------
    if (!ENABLE_ALL_FEATURES) {
        feature_mask = feature_masks[group_feature];
    }

    // -----------------------------------------------------------------------
    // Reduce the NUM_BANKS sub-histograms into one
    // -----------------------------------------------------------------------
    acc_type stat_val = 0.0f;
    uint cnt_val = 0;
    // 256 threads, working on 8 features and 16 bins, 2 stats
    // so each thread has an independent feature/bin/stat to work on.
    const ushort feature_id = ltid & DWORD_FEATURES_MASK; // bits 0-2, range 0-7
    ushort bin_id = ltid >> (LOG2_DWORD_FEATURES + 1);     // bits 4-7, range 0-15
    offset = (ltid >> (LOG2_DWORD_FEATURES + 1)) & BANK_MASK; // helps avoid LDS bank conflicts
    for (int i = 0; i < NUM_BANKS; ++i) {
        ushort bank_id = (i + offset) & BANK_MASK;
        stat_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 2 * DWORD_FEATURES + is_hessian_first * DWORD_FEATURES + feature_id];
    }
    if (CONST_HESSIAN) {
        if (ltid < LOCAL_SIZE_0 / 2) {
            // first 128 threads accumulate the 8 * 16 = 128 counter values
            bin_id = ltid >> LOG2_DWORD_FEATURES; // bits 3-6, range 0-15
            offset = (ltid >> LOG2_DWORD_FEATURES) & BANK_MASK;
            for (int i = 0; i < NUM_BANKS; ++i) {
                ushort bank_id = (i + offset) & BANK_MASK;
                cnt_val += cnt_hist[bin_id * CNT_BIN_MULT + bank_id * DWORD_FEATURES + feature_id];
            }
        }
    }

    // now thread  0- 7  holds feature 0-7's gradient for bin 0 and counter bin 0
    // now thread  8-15  holds feature 0-7's hessian  for bin 0 and counter bin 1
    // now thread 16-23  holds feature 0-7's gradient for bin 1 and counter bin 2
    // now thread 24-31  holds feature 0-7's hessian  for bin 1 and counter bin 3
    // etc.

    if (CONST_HESSIAN) {
        // Combine the two banks into one, and fill the Hessians with
        // counter_value * const_hessian
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gh_hist[ltid] = stat_val;
        if (ltid < LOCAL_SIZE_0 / 2) {
            cnt_hist[ltid] = cnt_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_hessian_first) {
            // these are the Hessians
            // thread  8-15 read counters stored by thread  0- 7
            // thread 24-31 read counters stored by thread  8-15
            // thread 40-47 read counters stored by thread 16-23, etc
            stat_val = const_hessian_val *
                       cnt_hist[((ltid - DWORD_FEATURES) >> (LOG2_DWORD_FEATURES + 1)) * DWORD_FEATURES + (ltid & DWORD_FEATURES_MASK)];
        }
        else {
            // these are the gradients
            // thread  0- 7 read gradients stored by thread  8-15
            // thread 16-23 read gradients stored by thread 24-31
            // thread 32-39 read gradients stored by thread 40-47, etc
            stat_val += gh_hist[ltid + DWORD_FEATURES];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -----------------------------------------------------------------------
    // Write to output / cross-workgroup reduction
    // -----------------------------------------------------------------------
    /* memory layout of output:
       g_f0_bin0   g_f1_bin0   ...  g_f7_bin0
       h_f0_bin0   h_f1_bin0   ...  h_f7_bin0
       g_f0_bin1   g_f1_bin1   ...  g_f7_bin1
       h_f0_bin1   h_f1_bin1   ...  h_f7_bin1
       ...
       g_f0_bin15  g_f1_bin15  ...  g_f7_bin15
       h_f0_bin15  h_f1_bin15  ...  h_f7_bin15
       c_f0_bin0   c_f1_bin0   ...  c_f7_bin0
       ...
       c_f0_bin15  c_f1_bin15  ...  c_f7_bin15
    */
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);

    if (POWER_FEATURE_WORKGROUPS != 0) {
        // Multiple workgroups per feature: write sub-histogram.
        // Matches original OpenCL format: direct ltid-indexed write.
        // ltid layout: [f0g_b0..f7g_b0, f0h_b0..f7h_b0, f0g_b1..f7g_b1, ...]
        device acc_type* output = (device acc_type*)output_buf + group_id * DWORD_FEATURES * 2 * NUM_BINS;
        output[ltid] = stat_val;
        // Sub-histograms written. CPU-side reduction follows.
    } else {
        // only 1 workgroup, no need to increase counter
        // the reduction will become a simple copy
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        device acc_type const* feature4_subhists =
                 (device acc_type*)output_buf + output_offset * DWORD_FEATURES * 2 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id_val = group_id ^ output_offset;
        // locate output histogram location for this feature4
        device acc_type* hist_buf = hist_buf_base + feature4_id * DWORD_FEATURES * 2 * NUM_BINS;
        within_kernel_reduction16x8(feature_mask, feature4_subhists, skip_id_val, stat_val,
                                    1 << POWER_FEATURE_WORKGROUPS, hist_buf, (threadgroup acc_type*)shared_array, ltid);
    }
}

// TODO: Add reduce_histogram16 kernel for multi-workgroup support once
// the histogram kernel's multi-threadgroup accumulation is debugged.
