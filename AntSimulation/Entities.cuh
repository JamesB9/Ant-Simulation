#pragma once

#include "Components.cuh"

typedef unsigned int EntityID;

struct Entities {

    // Arrays of data for each ant, sizes are all equal
    const unsigned int entityCount = 10000;
    PositionComponent* positions;
    MoveComponent* moves;
};
