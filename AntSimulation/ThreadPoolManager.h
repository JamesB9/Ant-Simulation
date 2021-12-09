///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: ThreadPoolManager.h
// Description: Header file for thread pools.
// Note: MADE REDUNDANT BY JAMES S. VERSION OF THREADING
// 
// Change Log:
//	-
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////

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
	thread* getThread(thread::id id);

	void queueJob(task job);

	int threadCount;
	bool anyJobs();
private:
	int jobsRunning = 0;
	condition_variable workers;
	mutex queue_mutex;
	queue<task> jobs;
	vector<thread> threads;
	map<thread::id, thread*> thread_map;
	map<thread::id, int> thread_task_map;

	
};