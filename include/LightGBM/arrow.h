/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <stdexcept>
#endif

/* -------------------------------------- C DATA INTERFACE ------------------------------------- */
// The C data interface is taken from
// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
// and is available under Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

#ifdef __cplusplus
extern "C" {
#endif

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#ifdef __cplusplus
}
#endif

/* --------------------------------------------------------------------------------------------- */
/*                                         CHUNKED ARRAY                                         */
/* --------------------------------------------------------------------------------------------- */

#ifdef __cplusplus
namespace LightGBM {

/**
 * @brief Arrow array-like container for a list of Arrow arrays.
 */
class ArrowChunkedArray {
  /* List of length `n` for `n` chunks containing the individual Arrow arrays. */
  std::vector<const ArrowArray*> chunks_;
  /* Schema for all chunks. */
  const ArrowSchema* schema_;
  /* List of length `n + 1` for `n` chunks containing the offsets for each chunk. */
  std::vector<int64_t> chunk_offsets_;
  /* Indicator whether this chunked array needs to call the arrays' release callbacks.
     NOTE: This is MUST only be set to `true` if this chunked array is not part of a
           `ArrowTable` as children arrays may not be released by the consumer (see below). */
  const bool releases_arrow_;

  inline void construct_chunk_offsets() {
    chunk_offsets_.reserve(chunks_.size() + 1);
    chunk_offsets_.emplace_back(0);
    for (size_t k = 0; k < chunks_.size(); ++k) {
      chunk_offsets_.emplace_back(chunks_[k]->length + chunk_offsets_.back());
    }
  }

 public:
  /**
   * @brief Construct a new Arrow Chunked Array object.
   *
   * @param chunks A list with the chunks.
   * @param schema The schema for all chunks.
   */
  inline ArrowChunkedArray(std::vector<const ArrowArray*> chunks, const ArrowSchema* schema)
      : releases_arrow_(false) {
    chunks_ = chunks;
    schema_ = schema;
    construct_chunk_offsets();
  }

  /**
   * @brief Construct a new Arrow Chunked Array object.
   *
   * @param n_chunks The number of chunks.
   * @param chunks A C-style array containing the chunks.
   * @param schema The schema for all chunks.
   */
  inline ArrowChunkedArray(int64_t n_chunks, const struct ArrowArray* chunks,
                           const struct ArrowSchema* schema)
      : releases_arrow_(true) {
    chunks_.reserve(n_chunks);
    for (int64_t k = 0; k < n_chunks; ++k) {
      if (chunks[k].length == 0) continue;
      chunks_.push_back(&chunks[k]);
    }
    schema_ = schema;
    construct_chunk_offsets();
  }

  ~ArrowChunkedArray() {
    if (!releases_arrow_) {
      return;
    }
    for (size_t i = 0; i < chunks_.size(); ++i) {
      auto chunk = chunks_[i];
      if (chunk->release) {
        chunk->release(const_cast<ArrowArray*>(chunk));
      }
    }
    if (schema_->release) {
      schema_->release(const_cast<ArrowSchema*>(schema_));
    }
  }

  /**
   * @brief Get the length of the chunked array.
   * This method returns the cumulative length of all chunks.
   * Complexity: O(1)
   *
   * @return int64_t The number of elements in the chunked array.
   */
  inline int64_t get_length() const { return chunk_offsets_.back(); }

  /* ----------------------------------------- ITERATOR ---------------------------------------- */
  template <typename T>
  class Iterator {
    using getter_fn = std::function<T(const ArrowArray*, int64_t)>;

    /* Reference to the chunked array that this iterator iterates over. */
    const ArrowChunkedArray& array_;
    /* Function to fetch the value at a certain index from a single chunk. */
    getter_fn get_;
    /* The chunk the iterator currently points to. */
    int64_t ptr_chunk_;
    /* The index inside the current chunk that the iterator points to. */
    int64_t ptr_offset_;

   public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = int64_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;

    /**
     * @brief Construct a new Iterator object.
     *
     * @param array Reference to the chunked array to iterator over.
     * @param get Function to fetch the value at a certain index from a single chunk.
     * @param ptr_chunk The index of the chunk to whose first index the iterator points to.
     */
    Iterator(const ArrowChunkedArray& array, getter_fn get, int64_t ptr_chunk);

    T operator*() const;
    template <typename I>
    T operator[](I idx) const;

    Iterator<T>& operator++();
    Iterator<T>& operator--();
    Iterator<T>& operator+=(int64_t c);

    template <typename V>
    friend bool operator==(const Iterator<V>& a, const Iterator<V>& b);
    template <typename V>
    friend bool operator!=(const Iterator<V>& a, const Iterator<V>& b);
    template <typename V>
    friend int64_t operator-(const Iterator<V>& a, const Iterator<V>& b);
  };

  /**
   * @brief Obtain an iterator to the beginning of the chunked array.
   *
   * @tparam T The value type of the iterator. May be any primitive type.
   * @return Iterator<T> The iterator.
   */
  template <typename T>
  inline Iterator<T> begin() const;

  /**
   * @brief Obtain an iterator to the beginning of the chunked array.
   *
   * @tparam T The value type of the iterator. May be any primitive type.
   * @return Iterator<T> The iterator.
   */
  template <typename T>
  inline Iterator<T> end() const;

  template <typename V>
  friend int64_t operator-(const Iterator<V>& a, const Iterator<V>& b);
};

/**
 * @brief Arrow container for a list of chunked arrays.
 */
class ArrowTable {
  std::vector<ArrowChunkedArray> columns_;
  const int64_t n_chunks_;
  const ArrowArray* chunks_ptr_;
  const ArrowSchema* schema_ptr_;

 public:
  /**
   * @brief Construct a new Arrow Table object.
   *
   * @param n_chunks The number of chunks.
   * @param chunks A C-style array containing the chunks.
   * @param schema The schema for all chunks.
   */
  inline ArrowTable(int64_t n_chunks, const ArrowArray* chunks, const ArrowSchema* schema)
      : n_chunks_(n_chunks), chunks_ptr_(chunks), schema_ptr_(schema) {
    columns_.reserve(schema->n_children);
    for (int64_t j = 0; j < schema->n_children; ++j) {
      std::vector<const ArrowArray*> children_chunks;
      children_chunks.reserve(n_chunks);
      for (int64_t k = 0; k < n_chunks; ++k) {
        if (chunks[k].length == 0) continue;
        children_chunks.push_back(chunks[k].children[j]);
      }
      columns_.emplace_back(children_chunks, schema->children[j]);
    }
  }

  ~ArrowTable() {
    // As consumer of the Arrow array, the Arrow table must release all Arrow arrays it receives
    // as well as the schema. As per the specification, children arrays are released by the
    // producer. See:
    // https://arrow.apache.org/docs/format/CDataInterface.html#release-callback-semantics-for-consumers
    for (int64_t i = 0; i < n_chunks_; ++i) {
      auto chunk = &chunks_ptr_[i];
      if (chunk->release) {
        chunk->release(const_cast<ArrowArray*>(chunk));
      }
    }
    if (schema_ptr_->release) {
      schema_ptr_->release(const_cast<ArrowSchema*>(schema_ptr_));
    }
  }

  /**
   * @brief Get the number of rows in the table.
   *
   * @return int64_t The number of rows.
   */
  inline int64_t get_num_rows() const { return columns_.front().get_length(); }

  /**
   * @brief Get the number of columns of this table.
   *
   * @return int64_t The column count.
   */
  inline int64_t get_num_columns() const { return columns_.size(); }

  /**
   * @brief Get the column at a particular index.
   *
   * @param idx The index of the column, must me in the range `[0, num_columns)`.
   * @return const ArrowChunkedArray& The chunked array for the child at the provided index.
   */
  inline const ArrowChunkedArray& get_column(size_t idx) const { return this->columns_[idx]; }
};

}  // namespace LightGBM

#include "arrow.tpp"
#endif /* __cplusplus */

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_
