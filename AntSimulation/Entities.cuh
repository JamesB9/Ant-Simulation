///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919),
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Entities.cuh
// Description: The header file for all of the Entities.
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
/// Headers
////////////////////////////////////////////////////////////
#pragma once
#include "Components.cuh"


////////////////////////////////////////////////////////////
/// \brief Entity Component System struct to hold arrays of component structs
///
/// \param entityCount Total number of entities and the size of all component arrays
/// \param moves Array of move components stored in GPU managed memory
/// \param sniffs Array of sniff components stored in GPU managed memory
/// \param activities Array of activity components stored in GPU managed memory
/// \param collisions Array of collision components stored in GPU managed memory
///
////////////////////////////////////////////////////////////
struct Entities {
    unsigned int entityCount;
    MoveComponent* moves;
    SniffComponent* sniffs;
    ActivityComponent* activities;
    CollisionComponent* collisions;
};
