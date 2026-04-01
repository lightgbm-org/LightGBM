/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_

#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/utils/array_args.h>

#include <string>
#include <cstdint>
#include <memory>
#include <vector>

#include "serial_tree_learner.h"

#ifdef LGBM_USE_METAL

namespace LightGBM {

/*!
 * \brief Metal GPU-based parallel learning algorithm for Apple Silicon.
 *        Accelerates histogram construction using Metal compute shaders.
 *        Split finding and tree construction remain on CPU.
 */
class MetalTreeLearner: public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* tree_config);
  ~MetalTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) override;
  void ResetIsConstantHessian(bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    SerialTreeLearner::SetBaggingData(subset, used_indices, num_data);
    if (subset == nullptr && used_indices != nullptr) {
      if (num_data != num_data_) {
        use_bagging_ = true;
        return;
      }
    }
    use_bagging_ = false;
  }

 protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void FindBestSplits(const Tree* tree) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

 private:
  /*! \brief 4-byte feature tuple used by GPU kernels */
  struct Feature4 {
      uint8_t s[4];
  };

  typedef float gpu_hist_t;

  /*!
   * \brief Find the best number of workgroups processing one feature
   * \param leaf_num_data The number of data examples on the current leaf
   * \return Log2 of the best number for workgroups per feature
   */
  int GetNumWorkgroupsPerFeature(data_size_t leaf_num_data);

  /*!
   * \brief Initialize Metal device, command queue, and load kernels
   */
  void InitMetal();

  /*!
   * \brief Allocate Metal shared-memory buffers
   */
  void AllocateMetalBuffers();

  /*!
   * \brief Load compiled metallib and create compute pipeline states
   */
  void BuildMetalKernels();

  /*!
   * \brief Compute GPU histogram for the current leaf
   * \param leaf_num_data Number of data on current leaf
   * \param use_all_features If true, skip feature masks for a faster path
   */
  void MetalHistogram(data_size_t leaf_num_data, bool use_all_features);

  /*!
   * \brief Wait for pending GPU command buffer to complete
   */
  void WaitForGPU();

  /*!
   * \brief Process GPU histogram results after GPU completion.
   *        Performs multi-workgroup reduction and bin redistribution.
   * \param histograms Destination of histogram results
   */
  void ProcessHistogramResults(hist_t* histograms);

  /*!
   * \brief Wait for GPU completion and read histogram results (convenience wrapper)
   * \param histograms Destination of histogram results
   */
  void WaitAndGetHistograms(hist_t* histograms);

  /*!
   * \brief Construct GPU histogram asynchronously
   * \param is_feature_used A predicate vector for enabling each feature
   * \param data_indices Array of data indices for current leaf
   * \param num_data Number of data examples
   * \param gradients Array of gradients for all examples
   * \param hessians Array of Hessians for all examples
   * \param ordered_gradients Destination for ordered gradients
   * \param ordered_hessians Destination for ordered Hessians
   * \return true if GPU kernel was launched
   */
  bool ConstructMetalHistogramsAsync(
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* data_indices, data_size_t num_data,
    const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians);

  /*! \brief Log2 of max number of workgroups per feature */
  static constexpr int kMaxLogWorkgroupsPerFeature = 10;
  /*! \brief Max total number of workgroups with preallocated workspace */
  int preallocd_max_num_wg_ = 1024;

  /*! \brief True if bagging is used */
  bool use_bagging_;

  // Metal objects stored as opaque pointers (Objective-C types in .mm)
  /*! \brief Metal device (id<MTLDevice>) */
  void* metal_device_ = nullptr;
  /*! \brief Metal command queue (id<MTLCommandQueue>) */
  void* metal_command_queue_ = nullptr;
  /*! \brief Metal library from compiled metallib (id<MTLLibrary>) */
  void* metal_library_ = nullptr;
  /*! \brief Pending command buffer for async execution (id<MTLCommandBuffer>) */
  void* pending_command_buffer_ = nullptr;

  // Pipeline state arrays stored as opaque pointers to NSMutableArray
  /*! \brief Pipeline states for standard kernels */
  void* histogram_pipelines_ = nullptr;
  /*! \brief Pipeline states for all-features-enabled kernels */
  void* histogram_allfeats_pipelines_ = nullptr;
  /*! \brief Pipeline states for full-data kernels */
  void* histogram_fulldata_pipelines_ = nullptr;

  // Metal buffers (id<MTLBuffer>) — unified memory, no separate host/device
  void* features_buffer_ = nullptr;     /*!< Feature4 packed training data */
  void* gradients_buffer_ = nullptr;    /*!< Ordered gradients */
  void* hessians_buffer_ = nullptr;     /*!< Ordered hessians */
  void* data_indices_buffer_ = nullptr; /*!< Data indices for current leaf */
  void* feature_masks_buffer_ = nullptr;/*!< Per-feature-group enable mask */
  void* subhistograms_buffer_ = nullptr;/*!< Temp sub-histograms for multi-WG reduction */
  void* histogram_output_buffer_ = nullptr; /*!< Final output histogram */

  /*! \brief Current exp_workgroups_per_feature for the pending dispatch */
  int pending_exp_workgroups_ = 0;

  /*! \brief total number of feature-groups */
  int num_feature_groups_;
  /*! \brief total number of dense feature-groups, processed on GPU */
  int num_dense_feature_groups_;
  /*! \brief Features per DWORD: 4 (bins>16) or 8 (bins<=16) */
  int dword_features_;
  /*! \brief Number of Feature4 tuples on GPU */
  int num_dense_feature4_;
  /*! \brief Max number of bins */
  int max_num_bin_;
  /*! \brief GPU kernel bin size (16, 64, 256) */
  int device_bin_size_;
  /*! \brief Size of histogram bin entry: sizeof(float)*2 */
  size_t hist_bin_entry_sz_;
  /*! \brief Indices of dense feature-groups */
  std::vector<int> dense_feature_group_map_;
  /*! \brief Indices of sparse feature-groups */
  std::vector<int> sparse_feature_group_map_;
  /*! \brief Multipliers of dense feature-groups for bin redistribution */
  std::vector<int> device_bin_mults_;
  /*! \brief Feature masks: 1=used, 0=not used */
  std::vector<char> feature_masks_;
  /*! \brief Which kernel to use: "histogram16", "histogram64", or "histogram256" */
  std::string kernel_name_;
};

}  // namespace LightGBM
#else

// When Metal support is not compiled in, quit with an error message

namespace LightGBM {

class MetalTreeLearner: public SerialTreeLearner {
 public:
  #ifdef _MSC_VER
    #pragma warning(disable : 4702)
  #endif
  explicit MetalTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("Metal Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DLGBM_USE_METAL=1");
  }
};

}  // namespace LightGBM

#endif   // LGBM_USE_METAL

#endif   // LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_
