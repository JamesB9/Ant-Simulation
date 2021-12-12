#pragma once
#define _USE_MATH_DEFINES
#include "math.h"

namespace Config {
	////////////// WORLD SIZE //////////////
	static const int WORLD_SIZE_X				= 800;
	static const int WORLD_SIZE_Y				= 800;

	////////////// PHEROMONES //////////////
	static const int ITEM_GRID_SIZE_X			= 160;
	static const int ITEM_GRID_SIZE_Y			= 160;
	static const float PHEROMONE_DECAY_STRENGH	= 10000 / 2000000.0f; // Pheromone removed from each cell per second
	static const float MAX_PHEROMONE_STORED_FOOD	= 50.0f; // Max amount of food per cell
	static const float MAX_PHEROMONE_STORED_HOME	= 50.0f; // Max home pheromone intensity per cell

	////////////// MAP //////////////
	static const int MAP_SIZE_X					= 80;
	static const int MAP_SIZE_Y					= 80;
	static const int MAP_SEED					= 111111111111; // -1 for random map

	//DEMO SEED: -1616234067

	////////////// ANTS //////////////
	static const int ANT_COUNT					= 10000;
	static const float ANT_MAX_SPEED			= 25.0f;
	static const float ANT_TURN_FORCE			= ANT_MAX_SPEED * 25.0f;
	static const float ANT_ROAM_STRENGTH		= 0.75f;
	static const float ANT_COLLISION_DISTANCE	= 5.0f;
	static const float ANT_COLLISION_FOV		= M_PI_4;
	static const int ANT_MAX_SNIFF_DISTANCE		= 5;
	static const float ANT_SNIFF_STRENGTH		= 5.0f;

	static const float INITIAL_DROP_STRENGTH	= 0.01f; // Max Pheromone drop strength
	static const float DROP_STRENGTH_REDUCTION  = 0.0001f; // Reduction in pheromone drop strength per second
	static const float PHEROMONE_DROP_TIME = 0.075f; // Amount of time (seconds) between each pheromone drop by an ant

	static const float PHEROMONE_DISPLAY_UPPER_BOUND = ((INITIAL_DROP_STRENGTH)*ANT_COUNT)/5.5;

	////////////// COLONIES //////////////
	static const int COLONY_COUNT = 1;
}
