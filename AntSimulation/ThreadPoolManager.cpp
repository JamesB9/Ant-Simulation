#include "ThreadPoolManager.h"

ThreadPoolManager::ThreadPoolManager()
{
	int threadCount = 10; //thread::hardware_concurrency();
	for (auto i = 0; i < threadCount; i++) {
		threads.push_back(thread(&ThreadPoolManager::queueWait, this));
	}
	
	/*for (int i = 0; i < threadCount; i++) {
		task t = { true, [i] { cout << "Test Job run on thread ID: " << this_thread::get_id() << endl; }};
		queueJob(t);
	}*/
}

void ThreadPoolManager::terminateThreads()
{
}

void ThreadPoolManager::queueWait()
{
	while (true) {
		unique_lock<mutex> lock(queue_mutex); //Lock access to only this thread
		if (!jobs.empty()) {
			//Grab first item from the job queue
			task Job = jobs.front();
			jobs.pop(); //Remove that item from the queue;

			//Do the job we picked up (Lock if asked to)
			if (!Job.lockNeeded) { lock.unlock(); }

			//cout << "Thread [#" << this_thread::get_id() << "] Starting Job" << endl;
			Job.func();

			if (Job.lockNeeded) { lock.unlock(); }
			
		}
		else {
			workers.wait(lock);
			continue;
		}
	}
}

void ThreadPoolManager::queueJob(task job)
{
	queue_mutex.lock();
	jobs.push(job);
	jobsTotal++;
	//cout << "Total Jobs: " << jobs.size() << endl;
	workers.notify_one();
	queue_mutex.unlock();
}
