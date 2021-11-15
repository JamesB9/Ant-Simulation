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
#pragma once

#include "Components.cuh"

typedef unsigned int EntityID;

struct Entities {

    // Arrays of data for each ant, sizes are all equal
    const unsigned int entityCount = 100000;
    MoveComponent* moves;
    SniffComponent* sniffs;
};
