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