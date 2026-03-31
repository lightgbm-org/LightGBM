/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_METAL

#include "metal_tree_learner.h"

#include <LightGBM/bin.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../io/dense_bin.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define METAL_DEBUG 0

namespace LightGBM {

// ============================================================================
// Lifecycle
// ============================================================================

MetalTreeLearner::MetalTreeLearner(const Config* config)
  : SerialTreeLearner(config) {
  use_bagging_ = false;
  Log::Info("Using Metal (Apple Silicon) tree learner");
}

MetalTreeLearner::~MetalTreeLearner() {
  // Release all retained Objective-C objects by transferring ownership back to ARC
  if (histogram_output_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)histogram_output_buffer_;
    histogram_output_buffer_ = nullptr;
  }
  if (sync_counters_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)sync_counters_buffer_;
    sync_counters_buffer_ = nullptr;
  }
  if (subhistograms_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)subhistograms_buffer_;
    subhistograms_buffer_ = nullptr;
  }
  if (feature_masks_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)feature_masks_buffer_;
    feature_masks_buffer_ = nullptr;
  }
  if (data_indices_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)data_indices_buffer_;
    data_indices_buffer_ = nullptr;
  }
  if (hessians_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)hessians_buffer_;
    hessians_buffer_ = nullptr;
  }
  if (gradients_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)gradients_buffer_;
    gradients_buffer_ = nullptr;
  }
  if (features_buffer_) {
    (void)(__bridge_transfer id<MTLBuffer>)features_buffer_;
    features_buffer_ = nullptr;
  }
  if (histogram_fulldata_pipelines_) {
    (void)(__bridge_transfer NSMutableArray*)histogram_fulldata_pipelines_;
    histogram_fulldata_pipelines_ = nullptr;
  }
  if (histogram_allfeats_pipelines_) {
    (void)(__bridge_transfer NSMutableArray*)histogram_allfeats_pipelines_;
    histogram_allfeats_pipelines_ = nullptr;
  }
  if (histogram_pipelines_) {
    (void)(__bridge_transfer NSMutableArray*)histogram_pipelines_;
    histogram_pipelines_ = nullptr;
  }
  if (pending_command_buffer_) {
    (void)(__bridge_transfer id<MTLCommandBuffer>)pending_command_buffer_;
    pending_command_buffer_ = nullptr;
  }
  if (metal_library_) {
    (void)(__bridge_transfer id<MTLLibrary>)metal_library_;
    metal_library_ = nullptr;
  }
  if (metal_command_queue_) {
    (void)(__bridge_transfer id<MTLCommandQueue>)metal_command_queue_;
    metal_command_queue_ = nullptr;
  }
  if (metal_device_) {
    (void)(__bridge_transfer id<MTLDevice>)metal_device_;
    metal_device_ = nullptr;
  }
}

// ============================================================================
// Init / Reset
// ============================================================================

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  // some additional variables needed for Metal trainer
  num_feature_groups_ = train_data_->num_feature_groups();
  // Initialize Metal device, command queue, kernels, and allocate buffers
  InitMetal();
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  num_feature_groups_ = train_data_->num_feature_groups();
  // GPU memory has to be reallocated because data may have changed
  AllocateMetalBuffers();
}

void MetalTreeLearner::ResetIsConstantHessian(bool is_constant_hessian) {
  if (is_constant_hessian != share_state_->is_constant_hessian) {
    SerialTreeLearner::ResetIsConstantHessian(is_constant_hessian);
    BuildMetalKernels();
  }
}

Tree* MetalTreeLearner::Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) {
  return SerialTreeLearner::Train(gradients, hessians, is_first_tree);
}

// ============================================================================
// GetNumWorkgroupsPerFeature
// ============================================================================

int MetalTreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {
  // We roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // Also guarantee that there are at least 2K examples per workgroup.
  // Currently using single workgroup per feature (POWER=0) for correctness.
  // Multi-workgroup support requires further debugging of the sub-histogram
  // accumulation with multiple threadgroups. The infrastructure is in place
  // (interleaved sub-histogram writes + CPU-side reduction), but the kernel
  // produces incorrect histograms when POWER > 0. This will be addressed in
  // a follow-up.
  (void)leaf_num_data;
  return 0;
}

// ============================================================================
// InitMetal — create device, command queue, determine kernel variant, build
// ============================================================================

void MetalTreeLearner::InitMetal() {
  // Get the max bin size, used for selecting best GPU kernel
  max_num_bin_ = 0;
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (train_data_->IsMultiGroup(i)) {
      continue;
    }
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureGroupNumBin(i));
  }

  // determine which kernel to use based on the max number of bins
  if (max_num_bin_ <= 16) {
    kernel_name_ = "histogram16";
    device_bin_size_ = 16;
    dword_features_ = 8;
  } else if (max_num_bin_ <= 64) {
    kernel_name_ = "histogram64";
    device_bin_size_ = 64;
    dword_features_ = 4;
  } else if (max_num_bin_ <= 256) {
    kernel_name_ = "histogram256";
    device_bin_size_ = 256;
    dword_features_ = 4;
  } else {
    Log::Fatal("bin size %d cannot run on Metal GPU", max_num_bin_);
  }

  // Suggest optimal bin sizes
  int max_num_bin_no_categorical = 0;
  int cur_feature_group = 0;
  bool categorical_feature_found = false;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const int feature_group = train_data_->Feature2Group(inner_feature_index);
    const BinMapper* feature_bin_mapper = train_data_->FeatureBinMapper(inner_feature_index);
    if (feature_bin_mapper->bin_type() == BinType::CategoricalBin) {
      categorical_feature_found = true;
    }
    if (feature_group != cur_feature_group || inner_feature_index == num_features_ - 1) {
      if (!categorical_feature_found) {
        max_num_bin_no_categorical = std::max(max_num_bin_no_categorical, train_data_->FeatureGroupNumBin(cur_feature_group));
      }
      categorical_feature_found = false;
      cur_feature_group = feature_group;
    }
  }
  if (max_num_bin_no_categorical == 65) {
    Log::Warning("Setting max_bin to 63 is suggested for best performance");
  }
  if (max_num_bin_no_categorical == 17) {
    Log::Warning("Setting max_bin to 15 is suggested for best performance");
  }

  // Create Metal device
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      Log::Fatal("No Metal-capable GPU device found");
    }
    Log::Info("Using Metal Device: %s", [[device name] UTF8String]);

    // Retain into opaque pointer
    metal_device_ = (__bridge_retained void*)device;

    // Create command queue
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      Log::Fatal("Failed to create Metal command queue");
    }
    metal_command_queue_ = (__bridge_retained void*)queue;

    // Load compiled metallib
    NSError* error = nil;
    id<MTLLibrary> library = nil;

    // Try several locations for default.metallib
    NSString* libPath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];

    if (!libPath || ![[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
      // Try relative to the executable
      NSString* execPath = [[NSProcessInfo processInfo] arguments][0];
      NSString* dir = [execPath stringByDeletingLastPathComponent];
      libPath = [dir stringByAppendingPathComponent:@"default.metallib"];
    }

    if (![[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
      // Try current working directory
      libPath = @"default.metallib";
    }

    if (![[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
      // Try the lib directory relative to the executable
      NSString* execPath = [[NSProcessInfo processInfo] arguments][0];
      NSString* dir = [execPath stringByDeletingLastPathComponent];
      libPath = [dir stringByAppendingPathComponent:@"lib/default.metallib"];
    }

    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
      NSURL* libURL = [NSURL fileURLWithPath:libPath];
      library = [device newLibraryWithURL:libURL error:&error];
      if (!library) {
        Log::Fatal("Failed to load Metal library from %s: %s",
                   [libPath UTF8String],
                   [[error localizedDescription] UTF8String]);
      }
      Log::Info("Loaded pre-compiled Metal library from %s", [libPath UTF8String]);
    } else {
      // Try the default library
      library = [device newDefaultLibrary];
    }

    // If no pre-compiled metallib found, compile from .metal source at runtime
    if (!library) {
      Log::Info("No pre-compiled metallib found, compiling Metal kernels from source...");
      // Try to find .metal source files
      NSString* kernelDir = nil;
      #ifdef LIGHTGBM_METAL_KERNEL_DIR
      kernelDir = @LIGHTGBM_METAL_KERNEL_DIR;
      #endif
      if (!kernelDir || ![[NSFileManager defaultManager] fileExistsAtPath:kernelDir]) {
        // Try relative to executable
        NSString* execPath2 = [[NSProcessInfo processInfo] arguments][0];
        NSString* dir2 = [execPath2 stringByDeletingLastPathComponent];
        kernelDir = [dir2 stringByAppendingPathComponent:@"src/treelearner/metal"];
      }
      // Compile each .metal file into a separate library, then use the one we need
      // We need kernel_name_ to be set first, but at this point we may not know it yet.
      // Compile the kernel file that matches our device_bin_size_.
      // Since InitMetal is called after we know max_num_bin_, select the right kernel.
      NSString* kernelFile = nil;
      if (max_num_bin_ <= 16) {
        kernelFile = @"histogram16.metal";
      } else if (max_num_bin_ <= 64) {
        kernelFile = @"histogram64.metal";
      } else {
        kernelFile = @"histogram256.metal";
      }
      NSString* path = [kernelDir stringByAppendingPathComponent:kernelFile];
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (src) {
          MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
          if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
          } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
          }
          library = [device newLibraryWithSource:src options:opts error:&error];
          if (!library) {
            Log::Fatal("Failed to compile Metal kernel %s: %s",
                       [kernelFile UTF8String],
                       [[error localizedDescription] UTF8String]);
          }
          Log::Info("Metal kernel %s compiled from source successfully", [kernelFile UTF8String]);
        } else {
          Log::Fatal("Could not read Metal kernel source %s: %s",
                     [path UTF8String], [[error localizedDescription] UTF8String]);
        }
      } else {
        Log::Fatal("Cannot find Metal kernel source at %s. "
                   "Ensure .metal files are available or build with Metal toolchain to create default.metallib",
                   [path UTF8String]);
      }
    }

    metal_library_ = (__bridge_retained void*)library;
    Log::Info("Metal library loaded successfully");
  }

  // Build pipeline states for all kernel variants
  BuildMetalKernels();
  // Allocate all GPU buffers and pack feature data
  AllocateMetalBuffers();
}

// ============================================================================
// BuildMetalKernels — create pipeline states for each power/variant combo
// ============================================================================

void MetalTreeLearner::BuildMetalKernels() {
  Log::Info("Building Metal compute pipelines for kernel '%s' with %d bins...",
            kernel_name_.c_str(), device_bin_size_);

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device_;
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)metal_library_;

    // Release old pipeline arrays if they exist
    if (histogram_pipelines_) {
      (void)(__bridge_transfer NSMutableArray*)histogram_pipelines_;
      histogram_pipelines_ = nullptr;
    }
    if (histogram_allfeats_pipelines_) {
      (void)(__bridge_transfer NSMutableArray*)histogram_allfeats_pipelines_;
      histogram_allfeats_pipelines_ = nullptr;
    }
    if (histogram_fulldata_pipelines_) {
      (void)(__bridge_transfer NSMutableArray*)histogram_fulldata_pipelines_;
      histogram_fulldata_pipelines_ = nullptr;
    }

    NSMutableArray* pipelines = [[NSMutableArray alloc] initWithCapacity:(kMaxLogWorkgroupsPerFeature + 1)];
    NSMutableArray* allfeats_pipelines = [[NSMutableArray alloc] initWithCapacity:(kMaxLogWorkgroupsPerFeature + 1)];
    NSMutableArray* fulldata_pipelines = [[NSMutableArray alloc] initWithCapacity:(kMaxLogWorkgroupsPerFeature + 1)];

    NSString* kernelFunctionName = [NSString stringWithUTF8String:kernel_name_.c_str()];

    for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
      NSError* error = nil;

      // ---- Standard kernel (indices + feature masks) ----
      {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        int32_t power_val = static_cast<int32_t>(i);
        bool const_hessian_val = share_state_->is_constant_hessian;
        bool enable_all_features_val = false;
        bool ignore_indices_val = false;
        [constants setConstantValue:&power_val type:MTLDataTypeInt atIndex:0];
        [constants setConstantValue:&const_hessian_val type:MTLDataTypeBool atIndex:1];
        [constants setConstantValue:&enable_all_features_val type:MTLDataTypeBool atIndex:2];
        [constants setConstantValue:&ignore_indices_val type:MTLDataTypeBool atIndex:3];

        id<MTLFunction> function = [library newFunctionWithName:kernelFunctionName
                                                constantValues:constants
                                                         error:&error];
        if (!function) {
          Log::Fatal("Failed to create Metal function '%s' (standard, power=%d): %s",
                     kernel_name_.c_str(), i,
                     error ? [[error localizedDescription] UTF8String] : "unknown error");
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
        if (!pipeline) {
          Log::Fatal("Failed to create Metal pipeline state (standard, power=%d): %s",
                     i, error ? [[error localizedDescription] UTF8String] : "unknown error");
        }
        [pipelines addObject:pipeline];
      }

      // ---- All-features kernel (no feature mask branching) ----
      {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        int32_t power_val = static_cast<int32_t>(i);
        bool const_hessian_val = share_state_->is_constant_hessian;
        bool enable_all_features_val = true;
        bool ignore_indices_val = false;
        [constants setConstantValue:&power_val type:MTLDataTypeInt atIndex:0];
        [constants setConstantValue:&const_hessian_val type:MTLDataTypeBool atIndex:1];
        [constants setConstantValue:&enable_all_features_val type:MTLDataTypeBool atIndex:2];
        [constants setConstantValue:&ignore_indices_val type:MTLDataTypeBool atIndex:3];

        id<MTLFunction> function = [library newFunctionWithName:kernelFunctionName
                                                constantValues:constants
                                                         error:&error];
        if (!function) {
          Log::Fatal("Failed to create Metal function '%s' (allfeats, power=%d): %s",
                     kernel_name_.c_str(), i,
                     error ? [[error localizedDescription] UTF8String] : "unknown error");
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
        if (!pipeline) {
          Log::Fatal("Failed to create Metal pipeline state (allfeats, power=%d): %s",
                     i, error ? [[error localizedDescription] UTF8String] : "unknown error");
        }
        [allfeats_pipelines addObject:pipeline];
      }

      // ---- Full-data kernel (all features + ignore indices — root node) ----
      {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        int32_t power_val = static_cast<int32_t>(i);
        bool const_hessian_val = share_state_->is_constant_hessian;
        bool enable_all_features_val = true;
        bool ignore_indices_val = true;
        [constants setConstantValue:&power_val type:MTLDataTypeInt atIndex:0];
        [constants setConstantValue:&const_hessian_val type:MTLDataTypeBool atIndex:1];
        [constants setConstantValue:&enable_all_features_val type:MTLDataTypeBool atIndex:2];
        [constants setConstantValue:&ignore_indices_val type:MTLDataTypeBool atIndex:3];

        id<MTLFunction> function = [library newFunctionWithName:kernelFunctionName
                                                constantValues:constants
                                                         error:&error];
        if (!function) {
          Log::Fatal("Failed to create Metal function '%s' (fulldata, power=%d): %s",
                     kernel_name_.c_str(), i,
                     error ? [[error localizedDescription] UTF8String] : "unknown error");
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                    error:&error];
        if (!pipeline) {
          Log::Fatal("Failed to create Metal pipeline state (fulldata, power=%d): %s",
                     i, error ? [[error localizedDescription] UTF8String] : "unknown error");
        }
        [fulldata_pipelines addObject:pipeline];
      }
    }

    histogram_pipelines_ = (__bridge_retained void*)pipelines;
    histogram_allfeats_pipelines_ = (__bridge_retained void*)allfeats_pipelines;
    histogram_fulldata_pipelines_ = (__bridge_retained void*)fulldata_pipelines;
  }

  Log::Info("Metal compute pipelines built successfully");
}

// ============================================================================
// AllocateMetalBuffers — create shared-memory buffers and pack Feature4 data
// ============================================================================

void MetalTreeLearner::AllocateMetalBuffers() {
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device_;

    num_dense_feature_groups_ = 0;
    for (int i = 0; i < num_feature_groups_; ++i) {
      if (!train_data_->IsMultiGroup(i)) {
        num_dense_feature_groups_++;
      }
    }
    // how many feature-group tuples we have
    num_dense_feature4_ = (num_dense_feature_groups_ + (dword_features_ - 1)) / dword_features_;
    // leave some safe margin for prefetching
    // 256 work-items per workgroup. Each work-item prefetches one tuple for that feature
    int allocated_num_data = num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature);
    // clear sparse/dense maps
    dense_feature_group_map_.clear();
    device_bin_mults_.clear();
    sparse_feature_group_map_.clear();
    // do nothing if no features can be processed on GPU
    if (!num_dense_feature_groups_) {
      Log::Warning("Metal GPU acceleration is disabled because no non-trivial dense features can be found");
      return;
    }

    // Release old feature buffer if it exists
    if (features_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)features_buffer_;
      features_buffer_ = nullptr;
    }

    // Allocate features buffer (Feature4 packed training data)
    uint64_t features_size = static_cast<uint64_t>(num_dense_feature4_) * num_data_ * sizeof(Feature4);
    id<MTLBuffer> featuresBuf = [device newBufferWithLength:features_size
                                                   options:MTLResourceStorageModeShared];
    if (!featuresBuf) {
      Log::Fatal("Failed to allocate Metal features buffer (%llu bytes)", features_size);
    }
    features_buffer_ = (__bridge_retained void*)featuresBuf;

    // make ordered_gradients and Hessians larger (including extra room for prefetching)
    ordered_gradients_.resize(allocated_num_data);
    ordered_hessians_.resize(allocated_num_data);

    // Release old gradient/hessian buffers
    if (gradients_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)gradients_buffer_;
      gradients_buffer_ = nullptr;
    }
    if (hessians_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)hessians_buffer_;
      hessians_buffer_ = nullptr;
    }

    // Allocate gradients and hessians buffers (shared memory — no pinning needed)
    id<MTLBuffer> gradBuf = [device newBufferWithLength:allocated_num_data * sizeof(score_t)
                                               options:MTLResourceStorageModeShared];
    if (!gradBuf) {
      Log::Fatal("Failed to allocate Metal gradients buffer");
    }
    gradients_buffer_ = (__bridge_retained void*)gradBuf;

    id<MTLBuffer> hessBuf = [device newBufferWithLength:allocated_num_data * sizeof(score_t)
                                                options:MTLResourceStorageModeShared];
    if (!hessBuf) {
      Log::Fatal("Failed to allocate Metal hessians buffer");
    }
    hessians_buffer_ = (__bridge_retained void*)hessBuf;

    // Allocate feature masks buffer
    feature_masks_.resize(num_dense_feature4_ * dword_features_);
    std::memset(feature_masks_.data(), 0, num_dense_feature4_ * dword_features_);

    if (feature_masks_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)feature_masks_buffer_;
      feature_masks_buffer_ = nullptr;
    }

    id<MTLBuffer> masksBuf = [device newBufferWithLength:num_dense_feature4_ * dword_features_
                                                options:MTLResourceStorageModeShared];
    if (!masksBuf) {
      Log::Fatal("Failed to allocate Metal feature masks buffer");
    }
    std::memset([masksBuf contents], 0, num_dense_feature4_ * dword_features_);
    feature_masks_buffer_ = (__bridge_retained void*)masksBuf;

    // Allocate data indices buffer
    if (data_indices_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)data_indices_buffer_;
      data_indices_buffer_ = nullptr;
    }

    id<MTLBuffer> indicesBuf = [device newBufferWithLength:allocated_num_data * sizeof(data_size_t)
                                                  options:MTLResourceStorageModeShared];
    if (!indicesBuf) {
      Log::Fatal("Failed to allocate Metal data indices buffer");
    }
    std::memset([indicesBuf contents], 0, allocated_num_data * sizeof(data_size_t));
    data_indices_buffer_ = (__bridge_retained void*)indicesBuf;

    // histogram bin entry size is always float*2 for Metal (no double precision)
    hist_bin_entry_sz_ = sizeof(gpu_hist_t) * 2;
    Log::Info("Size of histogram bin entry: %zu", hist_bin_entry_sz_);

    // Allocate sub-histograms buffer
    if (!subhistograms_buffer_) {
      uint64_t subhist_size = static_cast<uint64_t>(preallocd_max_num_wg_) * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
      id<MTLBuffer> subhistBuf = [device newBufferWithLength:subhist_size
                                                     options:MTLResourceStorageModeShared];
      if (!subhistBuf) {
        Log::Fatal("Failed to allocate Metal sub-histograms buffer");
      }
      subhistograms_buffer_ = (__bridge_retained void*)subhistBuf;
    }

    // Allocate sync counters buffer
    if (sync_counters_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)sync_counters_buffer_;
      sync_counters_buffer_ = nullptr;
    }
    {
      uint64_t sync_size = static_cast<uint64_t>(num_dense_feature4_) * sizeof(int);
      id<MTLBuffer> syncBuf = [device newBufferWithLength:sync_size
                                                  options:MTLResourceStorageModeShared];
      if (!syncBuf) {
        Log::Fatal("Failed to allocate Metal sync counters buffer");
      }
      std::memset([syncBuf contents], 0, sync_size);
      sync_counters_buffer_ = (__bridge_retained void*)syncBuf;
    }

    // Allocate histogram output buffer
    if (histogram_output_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)histogram_output_buffer_;
      histogram_output_buffer_ = nullptr;
    }
    {
      uint64_t output_size = static_cast<uint64_t>(num_dense_feature4_) * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
      id<MTLBuffer> outputBuf = [device newBufferWithLength:output_size
                                                    options:MTLResourceStorageModeShared];
      if (!outputBuf) {
        Log::Fatal("Failed to allocate Metal histogram output buffer");
      }
      histogram_output_buffer_ = (__bridge_retained void*)outputBuf;
    }

    // ====================================================================
    // Feature4 packing — pack feature data into the features buffer
    // ====================================================================
    int k = 0, copied_feature4 = 0;
    std::vector<int> dense_dword_ind(dword_features_);

    for (int i = 0; i < num_feature_groups_; ++i) {
      // looking for dword_features_ non-sparse feature-groups
      if (!train_data_->IsMultiGroup(i)) {
        dense_dword_ind[k] = i;
        // decide if we need to redistribute the bin
        double t = device_bin_size_ / static_cast<double>(train_data_->FeatureGroupNumBin(i));
        // multiplier must be a power of 2
        device_bin_mults_.push_back(static_cast<int>(round(pow(2, floor(log2(t))))));
        k++;
      } else {
        sparse_feature_group_map_.push_back(i);
      }
      // found dword_features_ dense groups — pack them
      if (k == dword_features_) {
        k = 0;
        for (int j = 0; j < dword_features_; ++j) {
          dense_feature_group_map_.push_back(dense_dword_ind[j]);
        }
        copied_feature4++;
      }
    }

    auto start_time = std::chrono::steady_clock::now();

    // Write packed Feature4 data directly to shared buffer contents
    Feature4* device_features_ptr = reinterpret_cast<Feature4*>([featuresBuf contents]);

    int nthreads = std::min(OMP_NUM_THREADS(), static_cast<int>(dense_feature_group_map_.size()) / dword_features_);
    nthreads = std::max(nthreads, 1);

    // Allocate temporary host buffers for parallel packing
    std::vector<Feature4*> host4_vecs(nthreads);
    for (int i = 0; i < nthreads; ++i) {
      host4_vecs[i] = reinterpret_cast<Feature4*>(malloc(num_data_ * sizeof(Feature4)));
    }

    // building Feature4 bundles; each thread handles dword_features_ features
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int i = 0; i < static_cast<int>(dense_feature_group_map_.size() / dword_features_); ++i) {
      int tid = omp_get_thread_num();
      Feature4* host4 = host4_vecs[tid];
      auto dense_ind = dense_feature_group_map_.begin() + i * dword_features_;
      auto dev_bin_mult = device_bin_mults_.begin() + i * dword_features_;

      if (dword_features_ == 8) {
        // one feature datapoint is 4 bits
        BinIterator* bin_iters[8];
        for (int s_idx = 0; s_idx < 8; ++s_idx) {
          bin_iters[s_idx] = train_data_->FeatureGroupIterator(dense_ind[s_idx]);
          if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[s_idx]) == 0) {
            Log::Fatal("Metal tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not", dense_ind[s_idx]);
          }
        }
        DenseBinIterator<uint8_t, true> iters[8] = {
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[0]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[1]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[2]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[3]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[4]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[5]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[6]),
          *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[7])};
        for (int j = 0; j < num_data_; ++j) {
          host4[j].s[0] = (uint8_t)((iters[0].RawGet(j) * dev_bin_mult[0] + ((j+0) & (dev_bin_mult[0] - 1)))
                        |((iters[1].RawGet(j) * dev_bin_mult[1] + ((j+1) & (dev_bin_mult[1] - 1))) << 4));
          host4[j].s[1] = (uint8_t)((iters[2].RawGet(j) * dev_bin_mult[2] + ((j+2) & (dev_bin_mult[2] - 1)))
                        |((iters[3].RawGet(j) * dev_bin_mult[3] + ((j+3) & (dev_bin_mult[3] - 1))) << 4));
          host4[j].s[2] = (uint8_t)((iters[4].RawGet(j) * dev_bin_mult[4] + ((j+4) & (dev_bin_mult[4] - 1)))
                        |((iters[5].RawGet(j) * dev_bin_mult[5] + ((j+5) & (dev_bin_mult[5] - 1))) << 4));
          host4[j].s[3] = (uint8_t)((iters[6].RawGet(j) * dev_bin_mult[6] + ((j+6) & (dev_bin_mult[6] - 1)))
                        |((iters[7].RawGet(j) * dev_bin_mult[7] + ((j+7) & (dev_bin_mult[7] - 1))) << 4));
        }
      } else if (dword_features_ == 4) {
        // one feature datapoint is one byte
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
          BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_ind[s_idx]);
          if (dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iter) != 0) {
            DenseBinIterator<uint8_t, false> iter = *static_cast<DenseBinIterator<uint8_t, false>*>(bin_iter);
            for (int j = 0; j < num_data_; ++j) {
              host4[j].s[s_idx] = (uint8_t)(iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1)));
            }
          } else if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
            DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
            for (int j = 0; j < num_data_; ++j) {
              host4[j].s[s_idx] = (uint8_t)(iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1)));
            }
          } else {
            Log::Fatal("Bug in Metal tree builder: only DenseBin and Dense4bitsBin are supported");
          }
        }
      } else {
        Log::Fatal("Bug in Metal tree builder: dword_features_ can only be 4 or 8");
      }

      // Copy to the shared Metal buffer — unified memory, direct memcpy
      std::memcpy(device_features_ptr + static_cast<uint64_t>(i) * num_data_,
                  host4, num_data_ * sizeof(Feature4));
    }

    // working on the remaining (less than dword_features_) feature groups
    if (k != 0) {
      Feature4* host4 = host4_vecs[0];
      if (dword_features_ == 8) {
        std::memset(host4, 0, num_data_ * sizeof(Feature4));
      }

      for (int i = 0; i < k; ++i) {
        if (dword_features_ == 8) {
          BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_dword_ind[i]);
          if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
            DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
            #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
            for (int j = 0; j < num_data_; ++j) {
              host4[j].s[i >> 1] |= (uint8_t)((iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                                  + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)))
                                 << ((i & 1) << 2));
            }
          } else {
            Log::Fatal("Metal tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not", dense_dword_ind[i]);
          }
        } else if (dword_features_ == 4) {
          BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_dword_ind[i]);
          if (dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iter) != 0) {
            DenseBinIterator<uint8_t, false> iter = *static_cast<DenseBinIterator<uint8_t, false>*>(bin_iter);
            #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
            for (int j = 0; j < num_data_; ++j) {
              host4[j].s[i] = (uint8_t)(iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                            + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)));
            }
          } else if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
            DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
            #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
            for (int j = 0; j < num_data_; ++j) {
              host4[j].s[i] = (uint8_t)(iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                            + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)));
            }
          } else {
            Log::Fatal("BUG in Metal tree builder: only DenseBin and Dense4bitsBin are supported");
          }
        } else {
          Log::Fatal("Bug in Metal tree builder: dword_features_ can only be 4 or 8");
        }
      }
      // fill the leftover features
      if (dword_features_ == 8) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (int j = 0; j < num_data_; ++j) {
          for (int i = k; i < dword_features_; ++i) {
            host4[j].s[i >> 1] |= (uint8_t)((j & 0xf) << ((i & 1) << 2));
          }
        }
      } else if (dword_features_ == 4) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (int j = 0; j < num_data_; ++j) {
          for (int i = k; i < dword_features_; ++i) {
            host4[j].s[i] = (uint8_t)j;
          }
        }
      }

      // Copy the last partial feature4 tuple
      std::memcpy(device_features_ptr + (num_dense_feature4_ - 1) * static_cast<uint64_t>(num_data_),
                  host4, num_data_ * sizeof(Feature4));

      for (int i = 0; i < k; ++i) {
        dense_feature_group_map_.push_back(dense_dword_ind[i]);
      }
    }

    // deallocate temporary host buffers
    for (int i = 0; i < nthreads; ++i) {
      free(host4_vecs[i]);
    }

    // data transfer time
    std::chrono::duration<double, std::milli> end_time = std::chrono::steady_clock::now() - start_time;
    Log::Info("%d dense feature groups (%.2f MB) written to Metal shared buffer in %f secs. %d sparse feature groups",
              static_cast<int>(dense_feature_group_map_.size()),
              ((dense_feature_group_map_.size() + (dword_features_ - 1)) / dword_features_) * num_data_ * sizeof(Feature4) / (1024.0 * 1024.0),
              end_time.count() * 1e-3,
              static_cast<int>(sparse_feature_group_map_.size()));
  }
}

// ============================================================================
// BeforeTrain — copy gradients/hessians to shared buffers for root node
// ============================================================================

void MetalTreeLearner::BeforeTrain() {
  // Copy initial full hessians and gradients to Metal shared buffers.
  // With unified memory this is just a memcpy to buffer.contents.
  if (!use_bagging_ && num_dense_feature_groups_) {
    @autoreleasepool {
      id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
      id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;

      if (!share_state_->is_constant_hessian) {
        std::memcpy([hessBuf contents], hessians_, num_data_ * sizeof(score_t));
      }
      std::memcpy([gradBuf contents], gradients_, num_data_ * sizeof(score_t));
    }
  }

  SerialTreeLearner::BeforeTrain();

  // use bagging
  if (data_partition_->leaf_count(0) != num_data_ && num_dense_feature_groups_) {
    @autoreleasepool {
      // On Metal, we copy indices, gradients and Hessians now
      const data_size_t* indices = data_partition_->indices();
      data_size_t cnt = data_partition_->leaf_count(0);

      // copy indices to Metal shared buffer
      id<MTLBuffer> indicesBuf = (__bridge id<MTLBuffer>)data_indices_buffer_;
      std::memcpy([indicesBuf contents], indices, cnt * sizeof(data_size_t));

      if (!share_state_->is_constant_hessian) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < cnt; ++i) {
          ordered_hessians_[i] = hessians_[indices[i]];
        }
        id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
        std::memcpy([hessBuf contents], ordered_hessians_.data(), cnt * sizeof(score_t));
      }

      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < cnt; ++i) {
        ordered_gradients_[i] = gradients_[indices[i]];
      }
      id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
      std::memcpy([gradBuf contents], ordered_gradients_.data(), cnt * sizeof(score_t));
    }
  }
}

// ============================================================================
// BeforeFindBestSplit — copy indices and ordered grads/hessians for smaller leaf
// ============================================================================

bool MetalTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  int smaller_leaf;
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // only have root
  if (right_leaf < 0) {
    smaller_leaf = -1;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
  } else {
    smaller_leaf = right_leaf;
  }

  // Copy indices, gradients and Hessians as early as possible
  if (smaller_leaf >= 0 && num_dense_feature_groups_) {
    @autoreleasepool {
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
      data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);

      // copy indices to Metal shared buffer
      id<MTLBuffer> indicesBuf = (__bridge id<MTLBuffer>)data_indices_buffer_;
      std::memcpy([indicesBuf contents], indices + begin, (end - begin) * sizeof(data_size_t));

      if (!share_state_->is_constant_hessian) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = begin; i < end; ++i) {
          ordered_hessians_[i - begin] = hessians_[indices[i]];
        }
        id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
        std::memcpy([hessBuf contents], ordered_hessians_.data(), (end - begin) * sizeof(score_t));
      }

      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        ordered_gradients_[i - begin] = gradients_[indices[i]];
      }
      id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
      std::memcpy([gradBuf contents], ordered_gradients_.data(), (end - begin) * sizeof(score_t));
    }
  }
  return SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);
}

// ============================================================================
// MetalHistogram — encode and commit compute commands for histogram kernel
// ============================================================================

void MetalTreeLearner::MetalHistogram(data_size_t leaf_num_data, bool use_all_features) {
  @autoreleasepool {
    int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
    int num_workgroups = (1 << exp_workgroups_per_feature) * num_dense_feature4_;
    #if METAL_DEBUG >= 1
    Log::Info("MetalHistogram: leaf_num_data=%d, POWER=%d, num_workgroups=%d, num_dense_feature4=%d",
              leaf_num_data, exp_workgroups_per_feature, num_workgroups, num_dense_feature4_);
    #endif

    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device_;

    // Reallocate sub-histograms buffer if needed
    if (num_workgroups > preallocd_max_num_wg_) {
      preallocd_max_num_wg_ = num_workgroups;
      Log::Info("Increasing preallocd_max_num_wg_ to %d for launching more workgroups", preallocd_max_num_wg_);
      if (subhistograms_buffer_) {
        (void)(__bridge_transfer id<MTLBuffer>)subhistograms_buffer_;
        subhistograms_buffer_ = nullptr;
      }
      uint64_t subhist_size = static_cast<uint64_t>(preallocd_max_num_wg_) * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
      id<MTLBuffer> subhistBuf = [device newBufferWithLength:subhist_size
                                                     options:MTLResourceStorageModeShared];
      if (!subhistBuf) {
        Log::Fatal("Failed to reallocate Metal sub-histograms buffer");
      }
      subhistograms_buffer_ = (__bridge_retained void*)subhistBuf;
    }

    // Select the appropriate pipeline state
    NSMutableArray* pipelineArray;
    if (leaf_num_data == num_data_) {
      pipelineArray = (__bridge NSMutableArray*)histogram_fulldata_pipelines_;
    } else if (use_all_features) {
      pipelineArray = (__bridge NSMutableArray*)histogram_allfeats_pipelines_;
    } else {
      pipelineArray = (__bridge NSMutableArray*)histogram_pipelines_;
    }
    id<MTLComputePipelineState> pipeline = pipelineArray[exp_workgroups_per_feature];

    // Create command buffer
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_command_queue_;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    // For multi-workgroup: clear sub-histograms buffer to avoid stale data from
    // previous iterations (different iterations may use different POWER values)
    if (exp_workgroups_per_feature > 0) {
      id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
      id<MTLBuffer> subhistBufLocal = (__bridge id<MTLBuffer>)subhistograms_buffer_;
      [blit fillBuffer:subhistBufLocal range:NSMakeRange(0, [subhistBufLocal length]) value:0];
      [blit endEncoding];
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Set buffer arguments
    id<MTLBuffer> featuresBuf = (__bridge id<MTLBuffer>)features_buffer_;
    id<MTLBuffer> masksBuf = (__bridge id<MTLBuffer>)feature_masks_buffer_;
    id<MTLBuffer> indicesBuf = (__bridge id<MTLBuffer>)data_indices_buffer_;
    id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
    id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
    id<MTLBuffer> subhistBuf = (__bridge id<MTLBuffer>)subhistograms_buffer_;
    id<MTLBuffer> syncBuf = (__bridge id<MTLBuffer>)sync_counters_buffer_;
    id<MTLBuffer> outputBuf = (__bridge id<MTLBuffer>)histogram_output_buffer_;

    uint32_t feature_size = static_cast<uint32_t>(num_data_);
    uint32_t num_data_arg = static_cast<uint32_t>(leaf_num_data);

    // Index 0: features buffer (offset to start of features data)
    [encoder setBuffer:featuresBuf offset:0 atIndex:0];
    // Index 1: feature masks
    [encoder setBuffer:masksBuf offset:0 atIndex:1];
    // Index 2: feature_size (number of examples, stride for Feature4 array)
    [encoder setBytes:&feature_size length:sizeof(uint32_t) atIndex:2];
    // Index 3: data indices
    [encoder setBuffer:indicesBuf offset:0 atIndex:3];
    // Index 4: num_data (number of data on this leaf)
    [encoder setBytes:&num_data_arg length:sizeof(uint32_t) atIndex:4];
    // Index 5: gradients
    [encoder setBuffer:gradBuf offset:0 atIndex:5];
    // Index 6: ordered_hessians (always bound; ignored by kernel when CONST_HESSIAN=true)
    [encoder setBuffer:hessBuf offset:0 atIndex:6];
    // Index 7: const_hessian_val (always bound; ignored by kernel when CONST_HESSIAN=false)
    float const_hessian = share_state_->is_constant_hessian ? static_cast<float>(hessians_[0]) : 0.0f;
    [encoder setBytes:&const_hessian length:sizeof(float) atIndex:7];
    // Index 8: output_buf (sub-histograms workspace)
    [encoder setBuffer:subhistBuf offset:0 atIndex:8];
    // Index 9: sync_counters
    [encoder setBuffer:syncBuf offset:0 atIndex:9];
    // Index 10: hist_buf_base (final histogram output)
    [encoder setBuffer:outputBuf offset:0 atIndex:10];

    // Dispatch main histogram kernel
    MTLSize threadgroups = MTLSizeMake(num_workgroups, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];

    // For multi-workgroup: reduction is done on CPU after waitUntilCompleted,
    // since the sub-histogram layout requires a separated→interleaved conversion
    // that is simpler and more debuggable on CPU. The GPU already did the heavy
    // work (histogram accumulation); the reduction is a simple O(num_wg * bins) sum.

    [cmdBuf commit];

    // Store the pending command buffer for later wait
    if (pending_command_buffer_) {
      (void)(__bridge_transfer id<MTLCommandBuffer>)pending_command_buffer_;
    }
    pending_command_buffer_ = (__bridge_retained void*)cmdBuf;
    pending_exp_workgroups_ = exp_workgroups_per_feature;
  }
}

// ============================================================================
// WaitAndGetHistograms — wait for GPU, read results from shared buffer
// ============================================================================

void MetalTreeLearner::WaitAndGetHistograms(hist_t* histograms) {
  @autoreleasepool {
    // Wait for the GPU to finish
    id<MTLCommandBuffer> cmdBuf = (__bridge id<MTLCommandBuffer>)pending_command_buffer_;
    [cmdBuf waitUntilCompleted];

    if ([cmdBuf status] == MTLCommandBufferStatusError) {
      Log::Fatal("Metal compute kernel failed: %s",
                 [[[cmdBuf error] localizedDescription] UTF8String]);
    }

    // For multi-workgroup dispatches, the GPU wrote sub-histograms in separated format.
    // We need to reduce them into the output buffer in interleaved format on CPU.
    id<MTLBuffer> outputBuf = (__bridge id<MTLBuffer>)histogram_output_buffer_;
    gpu_hist_t* hist_outputs = reinterpret_cast<gpu_hist_t*>([outputBuf contents]);

    if (pending_exp_workgroups_ > 0) {
      // Multi-workgroup: reduce sub-histograms on CPU.
      // Sub-histograms are in the same interleaved format as the final output,
      // so reduction is simple element-wise addition.
      id<MTLBuffer> subhistBuf = (__bridge id<MTLBuffer>)subhistograms_buffer_;
      const gpu_hist_t* sub_hist = reinterpret_cast<const gpu_hist_t*>([subhistBuf contents]);
      int num_sub = 1 << pending_exp_workgroups_;
      int elems_per_sub = dword_features_ * 2 * device_bin_size_;

      std::memset(hist_outputs, 0, num_dense_feature4_ * dword_features_ * device_bin_size_ * sizeof(gpu_hist_t) * 2);

      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int f4 = 0; f4 < num_dense_feature4_; ++f4) {
        gpu_hist_t* dst = hist_outputs + f4 * dword_features_ * device_bin_size_ * 2;
        for (int s = 0; s < num_sub; ++s) {
          const gpu_hist_t* src = sub_hist + (f4 * num_sub + s) * elems_per_sub;
          for (int e = 0; e < elems_per_sub; ++e) {
            dst[e] += src[e];
          }
        }
      }
    }

    #if METAL_DEBUG >= 1
    // Debug: check if histogram output has any non-zero values
    {
      size_t total_entries = num_dense_feature4_ * dword_features_ * device_bin_size_ * 2;
      double sum = 0.0;
      int nonzero = 0;
      for (size_t i = 0; i < total_entries; ++i) {
        if (hist_outputs[i] != 0.0f) {
          nonzero++;
          sum += std::fabs(hist_outputs[i]);
        }
      }
      Log::Info("Metal histogram output: %d non-zero values out of %lu, sum=%.6f",
                nonzero, total_entries, sum);
      // Print first few non-zero
      if (nonzero > 0) {
        int printed = 0;
        for (size_t i = 0; i < total_entries && printed < 10; ++i) {
          if (hist_outputs[i] != 0.0f) {
            Log::Info("  hist_output[%lu] = %f", i, hist_outputs[i]);
            printed++;
          }
        }
      }
    }
    #endif

    // Redistribute histogram bins back to original feature layout
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int i = 0; i < num_dense_feature_groups_; ++i) {
      if (!feature_masks_[i]) {
        continue;
      }
      int dense_group_index = dense_feature_group_map_[i];
      auto old_histogram_array = histograms + train_data_->GroupBinBoundary(dense_group_index) * 2;
      int bin_size = train_data_->FeatureGroupNumBin(dense_group_index);

      if (device_bin_mults_[i] == 1) {
        for (int j = 0; j < bin_size; ++j) {
          GET_GRAD(old_histogram_array, j) = GET_GRAD(hist_outputs, i * device_bin_size_ + j);
          GET_HESS(old_histogram_array, j) = GET_HESS(hist_outputs, i * device_bin_size_ + j);
        }
      } else {
        // values of this feature has been redistributed to multiple bins; need a reduction here
        int ind = 0;
        for (int j = 0; j < bin_size; ++j) {
          double sum_g = 0.0, sum_h = 0.0;
          for (int k = 0; k < device_bin_mults_[i]; ++k) {
            sum_g += GET_GRAD(hist_outputs, i * device_bin_size_ + ind);
            sum_h += GET_HESS(hist_outputs, i * device_bin_size_ + ind);
            ind++;
          }
          GET_GRAD(old_histogram_array, j) = sum_g;
          GET_HESS(old_histogram_array, j) = sum_h;
        }
      }
    }
  }
}

// ============================================================================
// ConstructMetalHistogramsAsync — prepare data and launch GPU kernel
// ============================================================================

bool MetalTreeLearner::ConstructMetalHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data,
  const score_t* gradients, const score_t* hessians,
  score_t* ordered_gradients, score_t* ordered_hessians) {

  if (num_data <= 0) {
    return false;
  }
  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    return false;
  }

  @autoreleasepool {
    // copy data indices if it is not null
    if (data_indices != nullptr && num_data != num_data_) {
      id<MTLBuffer> indicesBuf = (__bridge id<MTLBuffer>)data_indices_buffer_;
      std::memcpy([indicesBuf contents], data_indices, num_data * sizeof(data_size_t));
    }

    // generate and copy ordered_gradients if gradients is not null
    if (gradients != nullptr) {
      id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
      if (num_data != num_data_) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data; ++i) {
          ordered_gradients[i] = gradients[data_indices[i]];
        }
        std::memcpy([gradBuf contents], ordered_gradients, num_data * sizeof(score_t));
      } else {
        std::memcpy([gradBuf contents], gradients, num_data * sizeof(score_t));
      }
    }

    // generate and copy ordered_hessians if Hessians is not null
    if (hessians != nullptr && !share_state_->is_constant_hessian) {
      id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
      if (num_data != num_data_) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data; ++i) {
          ordered_hessians[i] = hessians[data_indices[i]];
        }
        std::memcpy([hessBuf contents], ordered_hessians, num_data * sizeof(score_t));
      } else {
        std::memcpy([hessBuf contents], hessians, num_data * sizeof(score_t));
      }
    }
  }

  // converted indices in is_feature_used to feature-group indices
  std::vector<int8_t> is_feature_group_used(num_feature_groups_, 0);
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1024) if (num_features_ >= 2048)
  for (int i = 0; i < num_features_; ++i) {
    if (is_feature_used[i]) {
      is_feature_group_used[train_data_->Feature2Group(i)] = 1;
    }
  }

  // construct the feature masks for dense feature-groups
  int used_dense_feature_groups = 0;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1024) reduction(+:used_dense_feature_groups) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      feature_masks_[i] = 1;
      ++used_dense_feature_groups;
    } else {
      feature_masks_[i] = 0;
    }
  }
  bool use_all_features = used_dense_feature_groups == num_dense_feature_groups_;
  // if no feature group is used, just return and do not use GPU
  if (used_dense_feature_groups == 0) {
    return false;
  }

  // if not all feature groups are used, we need to transfer the feature mask to GPU
  // otherwise, we will use a specialized GPU kernel with all feature groups enabled
  if (!use_all_features) {
    @autoreleasepool {
      id<MTLBuffer> masksBuf = (__bridge id<MTLBuffer>)feature_masks_buffer_;
      std::memcpy([masksBuf contents], feature_masks_.data(), num_dense_feature4_ * dword_features_);
    }
  }

  // All data have been prepared, now run the Metal kernel
  MetalHistogram(num_data, use_all_features);
  return true;
}

// ============================================================================
// ConstructHistograms — split features into sparse/dense, dispatch to GPU/CPU
// ============================================================================

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  std::vector<int8_t> is_dense_feature_used(num_features_, 0);

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (!is_feature_used[feature_index]) continue;
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(feature_index))) {
      is_sparse_feature_used[feature_index] = 1;
    } else {
      is_dense_feature_used[feature_index] = 1;
    }
  }

  // construct smaller leaf
  hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;

  // ConstructMetalHistogramsAsync will return true if there are available feature groups dispatched to GPU
  bool is_gpu_used = ConstructMetalHistogramsAsync(is_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf(),
    nullptr, nullptr,
    nullptr, nullptr);

  // then construct sparse features on CPU
  train_data_->ConstructHistograms<false, 0>(is_sparse_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    share_state_.get(),
    ptr_smaller_leaf_hist_data);

  // wait for GPU to finish, only if GPU is actually used
  if (is_gpu_used) {
    WaitAndGetHistograms(ptr_smaller_leaf_hist_data);
  }

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    hist_t* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - kHistOffset;

    is_gpu_used = ConstructMetalHistogramsAsync(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data());

    // then construct sparse features on CPU
    train_data_->ConstructHistograms<false, 0>(is_sparse_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(),
      ptr_larger_leaf_hist_data);

    // wait for GPU to finish, only if GPU is actually used
    if (is_gpu_used) {
      WaitAndGetHistograms(ptr_larger_leaf_hist_data);
    }
  }
}

// ============================================================================
// FindBestSplits — delegate to SerialTreeLearner
// ============================================================================

void MetalTreeLearner::FindBestSplits(const Tree* tree) {
  SerialTreeLearner::FindBestSplits(tree);
}

// ============================================================================
// Split — delegate to SerialTreeLearner with sanity checks
// ============================================================================

void MetalTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
  SerialTreeLearner::Split(tree, best_Leaf, left_leaf, right_leaf);
  if (Network::num_machines() == 1) {
    // do some sanity check for the Metal GPU algorithm
    if (best_split_info.left_count < best_split_info.right_count) {
      if ((best_split_info.left_count != smaller_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count != larger_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("Bug in Metal histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n",
                   best_split_info.left_count, best_split_info.right_count,
                   smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    } else {
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 best_split_info.right_sum_gradient,
                                 best_split_info.right_sum_hessian,
                                 best_split_info.right_output);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                best_split_info.left_sum_gradient,
                                best_split_info.left_sum_hessian,
                                best_split_info.left_output);
      if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count != smaller_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("Bug in Metal histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n",
                   best_split_info.left_count, best_split_info.right_count,
                   smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    }
  }
}

}  // namespace LightGBM

#endif  // USE_METAL
