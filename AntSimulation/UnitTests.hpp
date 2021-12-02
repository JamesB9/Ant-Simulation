#pragma once
#include <iostream>
#include <string>
#include <iomanip>

#define TEST_PASS 0
#define TEST_FAIL 1

typedef unsigned int TEST_RESULT;
typedef TEST_RESULT (*function_pointer)(void);

std::string resultToString(TEST_RESULT result) {
	return result == TEST_PASS ? "\x1B[92mPASS\x1B[00m" : "\x1B[91mFAIL\x1B[00m";
}

void runTest(std::string name, function_pointer function) {
	TEST_RESULT result = function();

	std::cout << "Unit Test: " << std::setw(30) << name  << " | Result: " << resultToString(result) << std::endl;
}