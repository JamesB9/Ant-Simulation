#pragma once
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>
#include <vector>;

struct Job {
    Job(std::function<void(void*)> function, void* args) : function{ function }, args{ args }{}
    std::function<void(void*)> function;
    void* args;
};


class ThreadPool {
public:
    const int NUMBER_OF_THREADS = std::thread::hardware_concurrency();
    ThreadPool();
    ~ThreadPool();
    void push(Job job);
    void done();
    void infiniteLoopFunc();
    void createThreads(ThreadPool* tp);
    void join(int i) {
        threadPool.at(i).join();
    }

private:
    std::queue<Job> jobQueue;
    std::mutex mutexLock;
    std::condition_variable dataCondition;
    std::atomic<bool> acceptFunctions;
    std::vector<std::thread> threadPool;




};