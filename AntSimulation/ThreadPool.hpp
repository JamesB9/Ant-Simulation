#pragma once
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>
#include <vector>;
class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();
    void push(std::function<void()> func);
    void done();
    void infinite_loop_func();
    void createThreads(ThreadPool* tp);
private:
    std::queue<std::function<void()>> m_function_queue;
    std::mutex m_lock;
    std::condition_variable m_data_condition;
    std::atomic<bool> m_accept_functions;

    std::vector<std::thread> threadPool;
    const int NUMBER_OF_THREADS = std::thread::hardware_concurrency();


};