/*
 * yamc_rwlock_sched.hpp
 *
 * MIT License
 *
 * Copyright (c) 2017 yohhoy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_RWLOCK_SCHED_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_RWLOCK_SCHED_HPP_

#include <cassert>
#include <cstddef>

/// default shared_mutex rwlock policy
#ifndef YAMC_RWLOCK_SCHED_DEFAULT
#define YAMC_RWLOCK_SCHED_DEFAULT yamc::rwlock::ReaderPrefer
#endif

namespace yamc {

/*
 * readers-writer locking policy for basic_shared_(timed)_mutex<RwLockPolicy>
 *
 * - yamc::rwlock::ReaderPrefer
 * - yamc::rwlock::WriterPrefer
 */
namespace rwlock {

/// Reader prefer scheduling
///
/// NOTE:
//    This policy might introduce "Writer Starvation" if readers continuously
//    hold shared lock. PThreads rwlock implementation in Linux use this
//    scheduling policy as default. (see also PTHREAD_RWLOCK_PREFER_READER_NP)
//
struct ReaderPrefer {
  static const std::size_t writer_mask = ~(~std::size_t(0u) >> 1);  // MSB 1bit
  static const std::size_t reader_mask = ~std::size_t(0u) >> 1;

  struct state {
    std::size_t rwcount = 0;
  };

  static void before_wait_wlock(const state&) {}
  static void after_wait_wlock(const state&) {}

  static bool wait_wlock(const state& s) { return (s.rwcount != 0); }

  static void acquire_wlock(state* s) {
    assert(!(s->rwcount & writer_mask));
    s->rwcount |= writer_mask;
  }

  static void release_wlock(state* s) {
    assert(s->rwcount & writer_mask);
    s->rwcount &= ~writer_mask;
  }

  static bool wait_rlock(const state& s) { return (s.rwcount & writer_mask) != 0; }

  static void acquire_rlock(state* s) {
    assert((s->rwcount & reader_mask) < reader_mask);
    ++(s->rwcount);
  }

  static bool release_rlock(state* s) {
    assert(0 < (s->rwcount & reader_mask));
    return (--(s->rwcount) == 0);
  }
};

/// Writer prefer scheduling
///
/// NOTE:
///   If there are waiting writer, new readers are blocked until all shared lock
///   are released,
//    and the writer thread can get exclusive lock in preference to blocked
//    reader threads. This policy might introduce "Reader Starvation" if writers
//    continuously request exclusive lock.
///   (see also PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
///
struct WriterPrefer {
  static const std::size_t locked = ~(~std::size_t(0u) >> 1);  // MSB 1bit
  static const std::size_t wait_mask = ~std::size_t(0u) >> 1;

  struct state {
    std::size_t nwriter = 0;
    std::size_t nreader = 0;
  };

  static void before_wait_wlock(state* s) {
    assert((s->nwriter & wait_mask) < wait_mask);
    ++(s->nwriter);
  }

  static bool wait_wlock(const state& s) {
    return ((s.nwriter & locked) || 0 < s.nreader);
  }

  static void after_wait_wlock(state* s) {
    assert(0 < (s->nwriter & wait_mask));
    --(s->nwriter);
  }

  static void acquire_wlock(state* s) {
    assert(!(s->nwriter & locked));
    s->nwriter |= locked;
  }

  static void release_wlock(state* s) {
    assert(s->nwriter & locked);
    s->nwriter &= ~locked;
  }

  static bool wait_rlock(const state& s) { return (s.nwriter != 0); }

  static void acquire_rlock(state* s) {
    assert(!(s->nwriter & locked));
    ++(s->nreader);
  }

  static bool release_rlock(state* s) {
    assert(0 < s->nreader);
    return (--(s->nreader) == 0);
  }
};

}  // namespace rwlock
}  // namespace yamc

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_RWLOCK_SCHED_HPP_
