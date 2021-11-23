#include "ThreadPool.hpp";

ThreadPool::ThreadPool() : jobQueue(), mutexLock(), dataCondition(), acceptFunctions(true){
}

ThreadPool::~ThreadPool(){
}

void ThreadPool::push(Job job){
    std::unique_lock<std::mutex> lock(mutexLock); //Locks the queue so there is no conflict.
    jobQueue.push(job); //Add the function to the que for the pool to tackel.
    lock.unlock();  //notification, so the consumer can take the lock.
    dataCondition.notify_one();
}

void ThreadPool::done(){
    std::unique_lock<std::mutex> lock(mutexLock);
    acceptFunctions = false;
    lock.unlock();
    dataCondition.notify_all();
    //notify all waiting threads.
}

void ThreadPool::infinite_loop_func(){
    std::function<void(void*)> function;
    void* args;
    while (true){
        {
            std::unique_lock<std::mutex> lock(mutexLock);
            dataCondition.wait(lock, [this]() {return !jobQueue.empty() || !acceptFunctions; });
            if (!acceptFunctions && jobQueue.empty()){
                //lock will be release automatically.
                //finish the thread loop and let it join in the main thread.
                return;
            }
            //Reads the funtion and the arguments from the job queue and then removes the job from the queue.
            function = jobQueue.front().function;
            args = jobQueue.front().args;
            jobQueue.pop();
            //release the lock
        }
        function(args);
    }
}

void ThreadPool::createThreads(ThreadPool* tp) {
    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        threadPool.push_back(std::thread(&ThreadPool::infinite_loop_func, tp));
    }
}