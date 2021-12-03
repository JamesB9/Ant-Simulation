#pragma once
#include "Utilities.cuh"
#include "UnitTests.hpp"

namespace UtilitiesUnitTests {

	// TEST 1: getDistance()
	TEST_RESULT test1() {
		// Setup
		Vec2f v1 = { 0, 0 };
		Vec2f v2 = { 3, 4 };
		float distance = getDistance(v1, v2);

		// Test
		if (distance == 5.0f) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 2: clamp()
	TEST_RESULT test2() {
		// Setup
		Vec2f v1 = { 500, 200 };
		v1 = clamp(v1, 300);

		// Test
		if (v1.x == 300) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 3: getAngle()
	TEST_RESULT test3() {
		// Setup
		Vec2f v1 = { 1.0f, 1.0f };
		Vec2f v2 = { 1.0f, 10000.0f };
		float angle = getAngle(v1, v2);

		// Test (within an 1000th)
		if (angle <= M_PI_4+(M_PI_4/1000.0f) && angle >= M_PI_4 - (M_PI_4 / 1000.0f)) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 4: normaliseRadian
	TEST_RESULT test4() {
		// Setup
		float angle = -M_PI_2; // -90 degrees
		float normalisedAngle = normaliseRadian(angle);

		

		// Test (within 1000th)
		if (normalisedAngle <= M_PI_2 + (M_PI_2 / 1000.0f) && normalisedAngle >= M_PI_2 - (M_PI_2 / 1000.0f)) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 5: normaliseSurface
	TEST_RESULT test5() {
		// Setup
		Vec2f v1 = { 500, 200 };
		Vec2f v2 = { 350, 100 };
		Vec2f normal = normaliseSurface(v2, v1);
		bool equals = normal.x == 150 && normal.y == 100;

		// Test
		if (equals) return TEST_PASS;
		else return TEST_FAIL;
	}
}