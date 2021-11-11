#pragma once

// 2D position: just x,y coordinates
struct PositionComponent
{
    float x, y;
};

struct MoveComponent
{
    float velx, vely, rotation;
};