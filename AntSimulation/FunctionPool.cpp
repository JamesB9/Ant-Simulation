#include "FunctionPool.hpp"
void FunctionPool::pushFuntion(std::function<void()> function) {
	std::unique_lock<std::mutex> fLock(lock);
	functionQueue.push(function);
	fLock.unlock();
	dataCondidtion.notify_one();
}

void FunctionPool::infinateLoop() {
	std::function<void()> function;
	while (true) {
		{
			std::unique_lock<std::mutex> lLock(lock);
			//Waits for a function to be added to the pool or the pool to be closed.
			dataCondidtion.wait(lLock, [this]() {
				return !functionQueue.empty() || !acceptFunctions;
				});
			//Breaks the function of there is nothing to be done. 
			if (!acceptFunctions && functionQueue.empty()) return;
			//Pop the first function in the queue and run it:
			function = functionQueue.front();
			functionQueue.pop();
			//Lock is relesed when the block closes.
		}
		function();
	}
}