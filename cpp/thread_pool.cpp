#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>

namespace minillm {

ThreadPool::ThreadPool() {
    const auto hardware = std::max(1U, std::thread::hardware_concurrency());
    active_threads_ = static_cast<std::int32_t>(hardware);
    workers_.reserve(hardware);
    for (unsigned int index = 0; index < hardware; ++index) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    available_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            available_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }
}

void ThreadPool::set_active_threads(std::int32_t threads) {
    if (threads <= 0) throw std::invalid_argument("threads must be positive");
    std::lock_guard lock(mutex_);
    active_threads_ = std::min<std::int32_t>(threads, static_cast<std::int32_t>(workers_.size()));
}

std::int32_t ThreadPool::active_threads() const {
    std::lock_guard lock(mutex_);
    return active_threads_;
}

void ThreadPool::parallel_for(
    std::int64_t begin, std::int64_t end, std::int64_t minimum_grain,
    const std::function<void(std::int64_t, std::int64_t)>& function
) {
    const std::int64_t size = end - begin;
    if (size <= 0) return;
    const auto desired = static_cast<std::int64_t>(active_threads());
    const auto chunks = std::min(desired, (size + minimum_grain - 1) / minimum_grain);
    if (chunks <= 1) {
        function(begin, end);
        return;
    }
    struct Completion {
        std::mutex mutex;
        std::condition_variable condition;
        std::atomic<std::int64_t> remaining{0};
        std::exception_ptr error;
    };
    auto completion = std::make_shared<Completion>();
    completion->remaining = chunks;
    const std::int64_t chunk_size = (size + chunks - 1) / chunks;
    {
        std::lock_guard lock(mutex_);
        for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
            const std::int64_t chunk_begin = begin + chunk * chunk_size;
            const std::int64_t chunk_end = std::min(end, chunk_begin + chunk_size);
            tasks_.emplace_back([completion, function, chunk_begin, chunk_end] {
                try {
                    function(chunk_begin, chunk_end);
                } catch (...) {
                    std::lock_guard error_lock(completion->mutex);
                    if (!completion->error) completion->error = std::current_exception();
                }
                if (--completion->remaining == 0) completion->condition.notify_one();
            });
        }
    }
    available_.notify_all();
    std::unique_lock wait_lock(completion->mutex);
    completion->condition.wait(wait_lock, [&] { return completion->remaining == 0; });
    if (completion->error) std::rethrow_exception(completion->error);
}

ThreadPool& global_thread_pool() {
    static ThreadPool pool;
    return pool;
}

}  // namespace minillm

