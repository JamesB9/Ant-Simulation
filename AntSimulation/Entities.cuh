#pragma once

#include "Components.cuh"

typedef unsigned int EntityID;

struct Entities {

    // Arrays of data for each ant, sizes are all equal
    const unsigned int entityCount = 1000;
    PositionComponent* positions;
    MoveComponent* moves;
};
