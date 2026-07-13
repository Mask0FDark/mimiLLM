#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mimillm {

class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void set_active_threads(std::int32_t threads);
    [[nodiscard]] std::int32_t active_threads() const;
    void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t minimum_grain,
                      const std::function<void(std::int64_t, std::int64_t)>& function);

private:
    void worker_loop();
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_{false};
    std::int32_t active_threads_{1};
};

ThreadPool& global_thread_pool();

}  // namespace mimillm

