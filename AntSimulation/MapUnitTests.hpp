#pragma once
#include "UnitTests.hpp"
#include "Map.cuh"

namespace MapUnitTests {
	// TEST 1: makeMapPointer()
	TEST_RESULT test1() {
		// Setup
		Map* map1 = makeMapPointer(50, 50);
		Map* map2 = makeMapPointer("Maps\\test_map_food_2.png");

		// Test
		bool map1Created = map1->height == 50;
		bool map2Created = map2->height == Config::MAP_SIZE_Y;

		if (map1Created && map2Created) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 2: getMapValueAt()
	TEST_RESULT test2() {
		// Setup
		Map* map = makeMapPointer(80, 80);
		map->map[92] = 1;

		// Test
		int mapValue = getMapValueAt(*map, 12, 1);
		if (mapValue == 1) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 3: setMapValueAt()
	TEST_RESULT test3() {
		// Setup
		Map* map = makeMapPointer(80, 80);
		setMapValueAt(*map, 12, 1, 1);

		// Test
		if (map->map[92] == 1) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 4: initBlankMap()
	TEST_RESULT test4() {
		// Setup
		Map* map = makeMapPointer(80, 80);
		map->map[92] == 1;
		initBlankMap(map, 80, 80);

		// Test
		if (map->map[92] == 0) return TEST_PASS;
		else return TEST_FAIL;
	}
}
