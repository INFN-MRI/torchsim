// Shared by every TorchSim CPU kernel: a worker pool that outlives the calls
// that use it, and the two questions of how many workers a piece of work can
// actually feed.
#ifndef TORCHSIM_THREADS_HPP
#define TORCHSIM_THREADS_HPP

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace torchsim {

// Workers outlive the call that first needs them, so a kernel pays for a
// handoff rather than for thread creation.
//
// A job hands slot 0 to the calling thread and the rest to the pool. Slots are
// numbered, not owned by any particular worker, so which worker runs a slot
// cannot affect a result: everything the kernels index by "thread" is really
// indexed by slot.
class WorkerPool {
public:
    static WorkerPool& instance() {
        // Deliberately never destroyed. An extension can be torn down with
        // threads still parked, and joining them there risks a deadlock at
        // interpreter exit for no benefit.
        static WorkerPool* const pool = new WorkerPool();
        return *pool;
    }

    void run(const unsigned int slots, const std::function<void(unsigned int)>& task) {
        if (slots <= 1) {
            task(0);
            return;
        }
        // One job at a time. A kernel already spreads across every core, so a
        // second caller gains nothing by interleaving with the first.
        const std::lock_guard<std::mutex> submission(submission_mutex_);
        {
            const std::lock_guard<std::mutex> guard(mutex_);
            while (workers_.size() + 1 < slots) {
                workers_.emplace_back([this] { serve(); });
            }
            task_ = &task;
            next_slot_ = 1;
            slot_limit_ = slots;
            outstanding_ = slots - 1;
        }
        ready_.notify_all();
        task(0);
        std::unique_lock<std::mutex> guard(mutex_);
        finished_.wait(guard, [this] { return outstanding_ == 0; });
        task_ = nullptr;
        slot_limit_ = 0;
    }

private:
    WorkerPool() = default;

    void serve() {
        while (true) {
            unsigned int slot = 0;
            const std::function<void(unsigned int)>* task = nullptr;
            {
                std::unique_lock<std::mutex> guard(mutex_);
                ready_.wait(guard, [this] { return next_slot_ < slot_limit_; });
                slot = next_slot_++;
                task = task_;
            }
            (*task)(slot);
            {
                const std::lock_guard<std::mutex> guard(mutex_);
                --outstanding_;
                if (outstanding_ == 0) {
                    finished_.notify_one();
                }
            }
        }
    }

    std::mutex submission_mutex_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::vector<std::thread> workers_;
    const std::function<void(unsigned int)>* task_ = nullptr;
    unsigned int next_slot_ = 0;
    unsigned int slot_limit_ = 0;
    unsigned int outstanding_ = 0;
};

// The processors this process may actually run on, which is not the same as the
// ones the machine has: a container or a taskset narrows the CPU set without
// changing what hardware_concurrency reports, and workers placed outside that
// set only contend for the ones inside it.
inline unsigned int usable_processors() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int usable = CPU_COUNT(&set);
        if (usable > 0) {
            return static_cast<unsigned int>(usable);
        }
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

// Even from a pool a worker has to be worth waking, so a problem that cannot
// give every slot a few work items runs faster with fewer slots. An explicit
// request is honoured as given, capped only by the work available.
constexpr std::int64_t MIN_WORK_PER_THREAD = 4;

inline unsigned int worker_count(
    const int requested, const std::int64_t work_count
) {
    const std::int64_t available = requested > 0
        ? static_cast<std::int64_t>(requested)
        : static_cast<std::int64_t>(usable_processors());
    const std::int64_t affordable = requested > 0
        ? work_count
        : work_count / MIN_WORK_PER_THREAD;
    const std::int64_t count =
        std::min(available, std::max<std::int64_t>(1, affordable));
    return static_cast<unsigned int>(std::max<std::int64_t>(1, count));
}

}  // namespace torchsim

#endif  // TORCHSIM_THREADS_HPP
