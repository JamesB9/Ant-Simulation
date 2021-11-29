#pragma once

namespace Config {
	////////////// WORLD SIZE //////////////
	static const int WORLD_SIZE_X				= 800;
	static const int WORLD_SIZE_Y				= 800;

	////////////// PHEROMONES //////////////
	static const int ITEM_GRID_SIZE_X			= 200;
	static const int ITEM_GRID_SIZE_Y			= 200;
	static const float PHEROMONE_DECAY_STRENGH	= 10000 / 3000000.0f; // Intensity
	static const float MAX_PHEROMONE_STORED_FOOD	= 50.0f;
	static const float MAX_PHEROMONE_STORED_HOME	= 50.0f;

	////////////// MAP //////////////
	static const int MAP_SIZE_X					= 80;
	static const int MAP_SIZE_Y					= 80;
	static const int MAP_SEED					= -1; // -1 for random map

	////////////// ANTS //////////////
	static const int ANT_COUNT					= 10000;
	static const float ANT_MAX_SPEED			= 25.0f;
	static const float ANT_TURN_FORCE			= ANT_MAX_SPEED * 1.25f;
	static const float ANT_ROAM_STRENGTH		= 0.25f;
	static const float ANT_COLLISION_DISTANCE	= 25.0f;
	static const int ANT_MAX_SNIFF_DISTANCE		= 5;
	static const float ANT_SNIFF_STRENGTH = 5.0f;

	static const float INITIAL_DROP_STRENGTH	= 0.01f; // Max Pheromone drop strength
	static const float DROP_STRENGTH_REDUCTION  = 0.0001f; // Reduction in pheromone drop strength per second
	static const float PHEROMONE_DROP_TIME = 0.075f; // Amount of time (seconds) between each pheromone drop by an ant

	////////////// COLONIES //////////////
	static const int COLONY_COUNT = 1;
}
