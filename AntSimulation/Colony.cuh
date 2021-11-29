////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

////////////////////////////////////////////////////////////
/// \brief Holds information for an ant colony
///
/// \param nestPositionX The x-coordinate location for the colonies nest
/// \param nestPositionY The y-coordinate location for the colonies nest
/// \param nestRadius The radius of the ant nest circle
/// \param totalFood A counter for the number of food brought back to the nest
/// \param antCount Number of ants in the colony
///
/// \see createColoniesArray
///
////////////////////////////////////////////////////////////
struct Colony{
	float nestPositionX;
	float nestPositionY;
	float nestRadius;
	int totalFood;
	int antCount;
};


////////////////////////////////////////////////////////////
/// \brief Creates an array of ant colonies on the GPU in managed memory
///
/// \param colonyCount The number of colonies in the array
/// 
/// \return pointer to an array of colonies
///
////////////////////////////////////////////////////////////
Colony* createColoniesArray(int colonyCount);
