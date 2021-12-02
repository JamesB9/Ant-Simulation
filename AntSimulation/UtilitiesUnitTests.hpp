#pragma once
#include "Utilities.cuh"
#include "UnitTests.hpp"

namespace UtilitiesUnitTests {

	// TEST 1: Intialising Item Grid on GPU
	TEST_RESULT test1() {
		// Setup
		Vec2f v1 = { 0, 0 };
		Vec2f v2 = { 3, 4 };
		float distance = getDistance(v1, v2);

		// Test
		if (distance == 5.0f) return TEST_PASS;
		else return TEST_FAIL;
	}
}