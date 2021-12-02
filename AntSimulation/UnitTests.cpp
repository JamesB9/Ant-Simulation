///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919),
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Main.cpp
// Description: This is the main driver file for the program.
//
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:
//	- (URL)
//
//
//////////////////////////// 80 columns wide //////////////////////////////////

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include "UnitTests.hpp"
#include "ItemGridUnitTests.hpp"
#include "UtilitiesUnitTests.hpp"
#include "MapUnitTests.hpp"

////////////////////////////////////////////////////////////
// Main Function
////////////////////////////////////////////////////////////
int main() {
	// Draw Header
	std::cout << "------------------" << std::endl;
	std::cout << "   UNIT TESTING" << std::endl;
	std::cout << "------------------" << std::endl;
	// Item Grid
	std::cout << "\nItem Grid Unit Tests:" << std::endl;
	runTest("initItemGrid()", ItemGridUnitTests::test1);
	runTest("getCell()", ItemGridUnitTests::test2);
	runTest("getCellIndex()", ItemGridUnitTests::test3);
	runTest("updateCell()", ItemGridUnitTests::test4);

	// Utilities
	std::cout << "\nUtilities Unit Tests:" << std::endl;
	runTest("getDistance()", UtilitiesUnitTests::test1);
	runTest("clamp()", UtilitiesUnitTests::test2);
	runTest("getAngle()", UtilitiesUnitTests::test3);
	runTest("isLeft()", UtilitiesUnitTests::test4);
	runTest("normaliseRadian()", UtilitiesUnitTests::test5);
	runTest("normaliseSurface()", UtilitiesUnitTests::test6);

	// Map
	std::cout << "\nMap Unit Tests:" << std::endl;
	runTest("makeMapPointer()", MapUnitTests::test1);
	runTest("getMapValueAt()", MapUnitTests::test2);
	runTest("setMapValueAt()", MapUnitTests::test3);
	runTest("initBlankMap()", MapUnitTests::test4);
	return 0;
}
