#include "ThreadPool.hpp"
#include <iostream>

void testFunc(void* args) {
	int i = *(int*)args;
	std::cout << "test"<< i << std::endl;
}


int main() {
	ThreadPool tp;
	tp.createThreads(&tp);

	for (int i = 0; i < 50; i++) {
		
		tp.push(Job(testFunc, (void*)&i));
	}

	tp.done();

	for (unsigned int i = 0; i < tp.NUMBER_OF_THREADS; i++) {
		tp.join(i);
	}

	return 0;
}

