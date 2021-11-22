#pragma once

namespace Config {
	////////////// WORLD SIZE //////////////
	static const int WORLD_SIZE_X				= 800;
	static const int WORLD_SIZE_Y				= 800;

	////////////// PHEROMONES //////////////
	static const int ITEM_GRID_SIZE_X			= 160;
	static const int ITEM_GRID_SIZE_Y			= 160;
	static const float PHEROMONE_DECAY_STRENGH	= 0.1f; // Intensity

	////////////// MAP //////////////
	static const int MAP_SIZE_X					= 80;
	static const int MAP_SIZE_Y					= 80;
	static const int MAP_SEED					= 1111; // -1 for random map

	////////////// ANTS //////////////
	static const int ANT_COUNT					= 10000;
	static const float ANT_MAX_SPEED			= 15.0f;
	static const float ANT_TURN_FORCE			= ANT_MAX_SPEED * 30.0f;
	static const float ANT_ROAM_STRENGTH		= 2.5f;
	static const float ANT_COLLISION_DISTANCE	= 25.0f;
	static const int ANT_MAX_SNIFF_DISTANCE		= 5;

	static const float INITIAL_DROP_STRENGTH	= 1.0f; // Max Pheromone drop strength
	static const float DROP_STRENGTH_REDUCTION  = 0.05f; // Reduction in pheromone drop strength per second
	static const float PHEROMONE_DROP_TIME = 0.1f; // Amount of time (seconds) between each pheromone drop by an ant
}