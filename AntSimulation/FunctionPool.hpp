#pragma once
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>


class FunctionPool {
public:
	FunctionPool();
	void pushFuntion(std::function<void()> function);
	void finshPool();
	void infinateLoop();

private:
	std::queue<std::function<void()>> functionQueue;
	std::mutex lock;
	std::condition_variable dataCondidtion;
	std::atomic<bool> acceptFunctions = true;


};