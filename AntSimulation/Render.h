///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Utilities.cuh
// Description: The item grid header file for the simulation.
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#pragma once
#include <stdlib.h>
#include <string>


//Pool all render.cpp's here

struct Map {
	int height;
	int width;
	int percentFill;

	int **map;
};

void initMap(Map& map);
void initMap(Map& map, int height, int width);

void printMap(Map& map);