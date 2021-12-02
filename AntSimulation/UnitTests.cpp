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

////////////////////////////////////////////////////////////
// Main Function
////////////////////////////////////////////////////////////
int main() {
	
	runTest("Item Grid Initialised", ItemGridUnitTests::test1);
	runTest("Utitilies getDistance()", UtilitiesUnitTests::test1);
	return 0;
}
