#pragma once
#include <queue>;
#include <vector>;
#include <thread>;
#include <shared_mutex>;
#include <functional>;
#include <iostream>;

using namespace std;

struct task {
	bool lockNeeded;
	function<void()> func;
};

class ThreadPoolManager {
public:
	ThreadPoolManager(); //initialise all threads
	void terminateThreads(); //kill all threads
	void queueWait(); //infinite queue checking function

	void queueJob(task job);

private:
	int jobsTotal = 0;
	condition_variable workers;
	mutex queue_mutex;
	queue<task> jobs;
	vector<thread> threads;
};