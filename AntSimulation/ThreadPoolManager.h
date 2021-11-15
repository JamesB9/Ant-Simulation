#pragma once
#include <queue>;
#include <vector>;
#include <thread>;
#include <shared_mutex>;
#include <functional>;
#include <iostream>;
#include <map>;
#include <string>;

using namespace std;

struct task {
	int t_id;
	bool lockNeeded;
	function<void()> func;
};

class ThreadPoolManager {
public:
	ThreadPoolManager(); //initialise all threads
	void join(); //join all threads
	void detach(); //detach all threads
	void queueWait(); //infinite queue checking function
	bool queueEmpty();
	thread* getThread(thread::id id);

	void queueJob(task job);

	int threadCount;

private:
	int jobsTotal = 0;
	condition_variable workers;
	mutex queue_mutex;
	queue<task> jobs;
	vector<thread> threads;
	map<thread::id, thread*> thread_map;
	map<thread::id, int> thread_task_map;
};