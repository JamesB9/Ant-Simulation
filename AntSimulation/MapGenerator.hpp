#pragma once

#include "Map.cuh"

//void initArray(Map* map);
void initBlankMap(Map* map, int height, int width);
void generateMap(Map& map);
void fillMap(Map& map);
void smoothMap(Map& map);
void floodFill(Map& map);
void arrayCopy(Map& from, Map& to);
int getNeighbourWallCount(Map& map, int x, int y, int delta);
void createMap(Map* map, int width, int height);
void printMap(Map& map);