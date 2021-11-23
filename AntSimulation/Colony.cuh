#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Colony{
	float nestPositionX;
	float nestPositionY;
	float nestRadius;
	int totalFood;
	int antCount;
};

Colony* createColoniesArray(int colonyCount);
