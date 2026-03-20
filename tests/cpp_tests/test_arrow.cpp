/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#include <LightGBM/arrow.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

using LightGBM::ArrowChunkedArray;
using LightGBM::ArrowTable;

/* --------------------------------------------------------------------------------------------- */
/*                                             UTILS                                             */
/* --------------------------------------------------------------------------------------------- */
// This code is copied and adapted from the official Arrow producer examples:
// https://arrow.apache.org/docs/format/CDataInterface.html#exporting-a-struct-float32-utf8-array

static void release_schema(struct ArrowSchema* schema) {
  // Free children
  if (schema->children) {
    for (int64_t i = 0; i < schema->n_children; ++i) {
      struct ArrowSchema* child = schema->children[i];
      if (child->release) {
        child->release(child);
      }
      free(child);
    }
    free(schema->children);
  }

  // Finalize
  schema->release = nullptr;
}

static void release_array(struct ArrowArray* array) {
  // Free children
  if (array->children) {
    for (int64_t i = 0; i < array->n_children; ++i) {
      struct ArrowArray* child = array->children[i];
      if (child->release) {
        child->release(child);
      }
      free(child);
    }
    free(array->children);
  }

  // Free buffers
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    if (array->buffers[i]) {
      free(const_cast<void*>(array->buffers[i]));
    }
  }
  free(array->buffers);

  // Finalize
  array->release = nullptr;
}

/* ------------------------------------------ PRODUCER ----------------------------------------- */

class ArrowChunkedArrayTest : public testing::Test {
 protected:
  void SetUp() override {}

  /* -------------------------------------- ARRAY CREATION ------------------------------------- */

  char* build_validity_bitmap(int64_t size, std::vector<int64_t> null_indices = {}) {
    if (null_indices.empty()) {
      return nullptr;
    }
    auto num_bytes = (size + 7) / 8;
    auto validity = static_cast<char*>(malloc(num_bytes * sizeof(char)));
    memset(validity, 0xff, num_bytes * sizeof(char));
    for (auto idx : null_indices) {
      validity[idx / 8] &= ~(1 << (idx % 8));
    }
    return validity;
  }

  ArrowArray build_primitive_array(void* data, int64_t size, int64_t offset,
                                   std::vector<int64_t> null_indices) {
    const void** buffers = (const void**)malloc(sizeof(void*) * 2);
    buffers[0] = build_validity_bitmap(size, null_indices);
    buffers[1] = data;

    ArrowArray arr;
    arr.length = size - offset;
    arr.null_count = static_cast<int64_t>(null_indices.size());
    arr.offset = offset;
    arr.n_buffers = 2;
    arr.n_children = 0;
    arr.buffers = buffers;
    arr.children = nullptr;
    arr.dictionary = nullptr;
    arr.release = &release_array;
    arr.private_data = nullptr;
    return arr;
  }

  template <typename T>
  ArrowArray create_primitive_array(const std::vector<T>& values, int64_t offset = 0,
                                    std::vector<int64_t> null_indices = {}) {
    // NOTE: Arrow arrays have 64-bit alignment but we can safely ignore this in tests
    auto buffer = static_cast<T*>(malloc(sizeof(T) * values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
      buffer[i] = values[i];
    }
    return build_primitive_array(buffer, values.size(), offset, null_indices);
  }

  ArrowArray create_primitive_array(const std::vector<bool>& values, int64_t offset = 0,
                                    std::vector<int64_t> null_indices = {}) {
    auto num_bytes = (values.size() + 7) / 8;
    auto buffer = static_cast<char*>(calloc(sizeof(char), num_bytes));
    for (size_t i = 0; i < values.size(); ++i) {
      // By using `calloc` above, we only need to set 'true' values
      if (values[i]) {
        buffer[i / 8] |= (1 << (i % 8));
      }
    }
    return build_primitive_array(buffer, values.size(), offset, null_indices);
  }

  ArrowArray created_nested_array(const std::vector<ArrowArray*>& arrays) {
    auto children = static_cast<ArrowArray**>(malloc(sizeof(ArrowArray*) * arrays.size()));
    for (size_t i = 0; i < arrays.size(); ++i) {
      auto child = static_cast<ArrowArray*>(malloc(sizeof(ArrowArray)));
      *child = *arrays[i];
      children[i] = child;
    }

    ArrowArray arr;
    arr.length = children[0]->length;
    arr.null_count = 0;
    arr.offset = 0;
    arr.n_buffers = 0;
    arr.n_children = static_cast<int64_t>(arrays.size());
    arr.buffers = nullptr;
    arr.children = children;
    arr.dictionary = nullptr;
    arr.release = &release_array;
    arr.private_data = nullptr;
    return arr;
  }

  /* ------------------------------------- SCHEMA CREATION ------------------------------------- */

  template <typename T>
  ArrowSchema create_primitive_schema() {
    std::logic_error("not implemented");
  }

  template <>
  ArrowSchema create_primitive_schema<float>() {
    ArrowSchema schema;
    schema.format = "f";
    schema.name = nullptr;
    schema.metadata = nullptr;
    schema.flags = 0;
    schema.n_children = 0;
    schema.children = nullptr;
    schema.dictionary = nullptr;
    schema.release = nullptr;
    schema.private_data = nullptr;
    return schema;
  }

  template <>
  ArrowSchema create_primitive_schema<bool>() {
    ArrowSchema schema;
    schema.format = "b";
    schema.name = nullptr;
    schema.metadata = nullptr;
    schema.flags = 0;
    schema.n_children = 0;
    schema.children = nullptr;
    schema.dictionary = nullptr;
    schema.release = nullptr;
    schema.private_data = nullptr;
    return schema;
  }

  ArrowSchema create_nested_schema(const std::vector<ArrowSchema*>& arrays) {
    auto children = static_cast<ArrowSchema**>(malloc(sizeof(ArrowSchema*) * arrays.size()));
    for (size_t i = 0; i < arrays.size(); ++i) {
      auto child = static_cast<ArrowSchema*>(malloc(sizeof(ArrowSchema)));
      *child = *arrays[i];
      children[i] = child;
    }

    ArrowSchema schema;
    schema.format = "+s";
    schema.name = nullptr;
    schema.metadata = nullptr;
    schema.flags = 0;
    schema.n_children = static_cast<int64_t>(arrays.size());
    schema.children = children;
    schema.dictionary = nullptr;
    schema.release = &release_schema;
    schema.private_data = nullptr;
    return schema;
  }
};

/* --------------------------------------------------------------------------------------------- */
/*                                             TESTS                                             */
/* --------------------------------------------------------------------------------------------- */

TEST_F(ArrowChunkedArrayTest, GetLength) {
  auto schema = create_primitive_schema<float>();

  std::vector<float> dat1 = {1, 2};
  auto arr1 = create_primitive_array(dat1);
  ArrowChunkedArray ca1(1, &arr1, &schema);
  ASSERT_EQ(ca1.get_length(), 2);

  std::vector<float> dat2 = {3, 4, 5, 6};
  auto arr2 = create_primitive_array(dat1);
  auto arr3 = create_primitive_array(dat2);
  ArrowArray arrs[2] = {arr2, arr3};
  ArrowChunkedArray ca2(2, arrs, &schema);
  ASSERT_EQ(ca2.get_length(), 6);

  std::vector<bool> dat3 = {true, false, true, true};
  auto arr4 = create_primitive_array(dat3, 1);
  ArrowChunkedArray ca3(1, &arr4, &schema);
  ASSERT_EQ(ca3.get_length(), 3);
}

TEST_F(ArrowChunkedArrayTest, GetColumns) {
  std::vector<float> dat1 = {1, 2, 3};
  auto arr1 = create_primitive_array(dat1);
  std::vector<float> dat2 = {4, 5, 6};
  auto arr2 = create_primitive_array(dat2);
  std::vector<ArrowArray*> arrs = {&arr1, &arr2};
  auto arr = created_nested_array(arrs);

  auto schema1 = create_primitive_schema<float>();
  auto schema2 = create_primitive_schema<float>();
  std::vector<ArrowSchema*> schemas = {&schema1, &schema2};
  auto schema = create_nested_schema(schemas);

  ArrowTable table(1, &arr, &schema);
  ASSERT_EQ(table.get_num_rows(), 3);
  ASSERT_EQ(table.get_num_columns(), 2);

  auto ca1 = table.get_column(0);
  ASSERT_EQ(ca1.get_length(), 3);
  ASSERT_EQ(*ca1.begin<int32_t>(), 1);

  auto ca2 = table.get_column(1);
  ASSERT_EQ(ca2.get_length(), 3);
  ASSERT_EQ(*ca2.begin<int32_t>(), 4);
}

TEST_F(ArrowChunkedArrayTest, IteratorArithmetic) {
  std::vector<float> dat1 = {1, 2};
  auto arr1 = create_primitive_array(dat1);
  std::vector<float> dat2 = {3, 4, 5, 6};
  auto arr2 = create_primitive_array(dat2);
  std::vector<float> dat3 = {7};
  auto arr3 = create_primitive_array(dat3);
  auto schema = create_primitive_schema<float>();

  ArrowArray arrs[3] = {arr1, arr2, arr3};
  ArrowChunkedArray ca(3, arrs, &schema);

  // Arithmetic
  auto it = ca.begin<int32_t>();
  ASSERT_EQ(*it, 1);
  ++it;
  ASSERT_EQ(*it, 2);
  ++it;
  ASSERT_EQ(*it, 3);
  it += 2;
  ASSERT_EQ(*it, 5);
  it += 2;
  ASSERT_EQ(*it, 7);
  --it;
  ASSERT_EQ(*it, 6);

  // Subscripts
  ASSERT_EQ(it[0], 1);
  ASSERT_EQ(it[1], 2);
  ASSERT_EQ(it[2], 3);
  ASSERT_EQ(it[6], 7);

  // End
  auto end = ca.end<int32_t>();
  ASSERT_EQ(end - it, 2);
  ASSERT_EQ(end - ca.begin<int32_t>(), 7);
}

TEST_F(ArrowChunkedArrayTest, BooleanIterator) {
  std::vector<bool> dat1 = {false, true, false};
  auto arr1 = create_primitive_array(dat1, 0, {2});
  std::vector<bool> dat2 = {false, false, false, false, true, true, true, true, false, true};
  auto arr2 = create_primitive_array(dat2, 1);
  auto schema = create_primitive_schema<bool>();

  ArrowArray arrs[2] = {arr1, arr2};
  ArrowChunkedArray ca(2, arrs, &schema);

  // Check for values in first chunk
  auto it = ca.begin<float>();
  ASSERT_EQ(*it, 0);
  ASSERT_EQ(*(++it), 1);
  ASSERT_TRUE(std::isnan(*(++it)));

  // Check for some values in second chunk
  ASSERT_EQ(*(++it), 0);
  it += 3;
  ASSERT_EQ(*it, 1);
  it += 4;
  ASSERT_EQ(*it, 0);
  ASSERT_EQ(*(++it), 1);

  // Check end
  ASSERT_EQ(++it, ca.end<float>());
}

TEST_F(ArrowChunkedArrayTest, OffsetAndValidity) {
  std::vector<float> dat = {0, 1, 2, 3, 4, 5, 6};
  auto arr = create_primitive_array(dat, 2, {2, 3});
  auto schema = create_primitive_schema<float>();
  ArrowChunkedArray ca(1, &arr, &schema);

  auto it = ca.begin<double>();
  ASSERT_TRUE(std::isnan(*it));
  ASSERT_TRUE(std::isnan(*(++it)));
  ASSERT_EQ(it[2], 4);
  ASSERT_EQ(it[4], 6);

  arr.release(&arr);
}
