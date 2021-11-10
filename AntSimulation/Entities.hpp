#pragma once

#include <vector>
#include "Components.hpp"

typedef unsigned int EntityID;

struct Entities {

    // Arrays of data for each ant, sizes are all equal
    unsigned int entityCount = 0;
    std::vector<PositionComponent> positions;
    std::vector<MoveComponent> moves;
    std::vector<int> flags; // [IMPLEMENT in FUTURE] bit flags to determine if entity has component

    EntityID addEntity()
    {
        EntityID id = entityCount++;

        positions.push_back(PositionComponent());
        moves.push_back(MoveComponent());

        return id;
    }
};
