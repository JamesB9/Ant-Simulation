#include "ItemGrid.cuh"
#include "UnitTests.hpp"

namespace ItemGridUnitTests {

	// TEST 1: Intialising Item Grid on GPU
	TEST_RESULT test1() {
		// Setup
		ItemGrid* itemGrid = initItemGrid(100, 100);
		// Test
		bool isCreated = itemGrid->totalCells == 100 * 100;
		if (isCreated) return TEST_PASS;
		else return TEST_FAIL;
	}
}