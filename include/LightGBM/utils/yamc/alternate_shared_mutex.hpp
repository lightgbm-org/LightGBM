/*
 * alternate_shared_mutex.hpp
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
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_ALTERNATE_SHARED_MUTEX_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_ALTERNATE_SHARED_MUTEX_HPP_

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "yamc_rwlock_sched.hpp"

namespace yamc {

/*
 * alternate implementation of shared mutex variants
 *
 * - yamc::alternate::shared_mutex
 * - yamc::alternate::shared_timed_mutex
 * - yamc::alternate::basic_shared_mutex<RwLockPolicy>
 * - yamc::alternate::basic_shared_timed_mutex<RwLockPolicy>
 */
namespace alternate {

namespace detail {

template <typename RwLockPolicy>
class shared_mutex_base {
 protected:
  typename RwLockPolicy::state state_;
  std::condition_variable cv_;
  std::mutex mtx_;

  void lock() {
    std::unique_lock<decltype(mtx_)> lk(mtx_);
    RwLockPolicy::before_wait_wlock(state_);
    while (RwLockPolicy::wait_wlock(state_)) {
      cv_.wait(lk);
    }
    RwLockPolicy::after_wait_wlock(state_);
    RwLockPolicy::acquire_wlock(&state_);
  }

  bool try_lock() {
    std::lock_guard<decltype(mtx_)> lk(mtx_);
    if (RwLockPolicy::wait_wlock(state_)) return false;
    RwLockPolicy::acquire_wlock(state_);
    return true;
  }

  void unlock() {
    std::lock_guard<decltype(mtx_)> lk(mtx_);
    RwLockPolicy::release_wlock(&state_);
    cv_.notify_all();
  }

  void lock_shared() {
    std::unique_lock<decltype(mtx_)> lk(mtx_);
    while (RwLockPolicy::wait_rlock(state_)) {
      cv_.wait(lk);
    }
    RwLockPolicy::acquire_rlock(&state_);
  }

  bool try_lock_shared() {
    std::lock_guard<decltype(mtx_)> lk(mtx_);
    if (RwLockPolicy::wait_rlock(state_)) return false;
    RwLockPolicy::acquire_rlock(state_);
    return true;
  }

  void unlock_shared() {
    std::lock_guard<decltype(mtx_)> lk(mtx_);
    if (RwLockPolicy::release_rlock(&state_)) {
      cv_.notify_all();
    }
  }
};

}  // namespace detail

template <typename RwLockPolicy>
class basic_shared_mutex : private detail::shared_mutex_base<RwLockPolicy> {
  using base = detail::shared_mutex_base<RwLockPolicy>;

 public:
  basic_shared_mutex() = default;
  ~basic_shared_mutex() = default;

  basic_shared_mutex(const basic_shared_mutex&) = delete;
  basic_shared_mutex& operator=(const basic_shared_mutex&) = delete;

  using base::lock;
  using base::try_lock;
  using base::unlock;

  using base::lock_shared;
  using base::try_lock_shared;
  using base::unlock_shared;
};

using shared_mutex = basic_shared_mutex<YAMC_RWLOCK_SCHED_DEFAULT>;

template <typename RwLockPolicy>
class basic_shared_timed_mutex
    : private detail::shared_mutex_base<RwLockPolicy> {
  using base = detail::shared_mutex_base<RwLockPolicy>;

  using base::cv_;
  using base::mtx_;
  using base::state_;

  template <typename Clock, typename Duration>
  bool do_try_lockwait(const std::chrono::time_point<Clock, Duration>& tp) {
    std::unique_lock<decltype(mtx_)> lk(mtx_);
    RwLockPolicy::before_wait_wlock(state_);
    while (RwLockPolicy::wait_wlock(state_)) {
      if (cv_.wait_until(lk, tp) == std::cv_status::timeout) {
        if (!RwLockPolicy::wait_wlock(state_))  // re-check predicate
          break;
        RwLockPolicy::after_wait_wlock(state_);
        return false;
      }
    }
    RwLockPolicy::after_wait_wlock(state_);
    RwLockPolicy::acquire_wlock(state_);
    return true;
  }

  template <typename Clock, typename Duration>
  bool do_try_lock_sharedwait(
      const std::chrono::time_point<Clock, Duration>& tp) {
    std::unique_lock<decltype(mtx_)> lk(mtx_);
    while (RwLockPolicy::wait_rlock(state_)) {
      if (cv_.wait_until(lk, tp) == std::cv_status::timeout) {
        if (!RwLockPolicy::wait_rlock(state_))  // re-check predicate
          break;
        return false;
      }
    }
    RwLockPolicy::acquire_rlock(state_);
    return true;
  }

 public:
  basic_shared_timed_mutex() = default;
  ~basic_shared_timed_mutex() = default;

  basic_shared_timed_mutex(const basic_shared_timed_mutex&) = delete;
  basic_shared_timed_mutex& operator=(const basic_shared_timed_mutex&) = delete;

  using base::lock;
  using base::try_lock;
  using base::unlock;

  template <typename Rep, typename Period>
  bool try_lock_for(const std::chrono::duration<Rep, Period>& duration) {
    const auto tp = std::chrono::steady_clock::now() + duration;
    return do_try_lockwait(tp);
  }

  template <typename Clock, typename Duration>
  bool try_lock_until(const std::chrono::time_point<Clock, Duration>& tp) {
    return do_try_lockwait(tp);
  }

  using base::lock_shared;
  using base::try_lock_shared;
  using base::unlock_shared;

  template <typename Rep, typename Period>
  bool try_lock_shared_for(const std::chrono::duration<Rep, Period>& duration) {
    const auto tp = std::chrono::steady_clock::now() + duration;
    return do_try_lock_sharedwait(tp);
  }

  template <typename Clock, typename Duration>
  bool try_lock_shared_until(
      const std::chrono::time_point<Clock, Duration>& tp) {
    return do_try_lock_sharedwait(tp);
  }
};

using shared_timed_mutex = basic_shared_timed_mutex<YAMC_RWLOCK_SCHED_DEFAULT>;

}  // namespace alternate
}  // namespace yamc

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_ALTERNATE_SHARED_MUTEX_HPP_
