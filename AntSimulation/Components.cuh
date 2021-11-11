
#define LEAVING_HOME 0;
#define FOUND_FOOD 1;

struct PositionComponent
{
    float x, y;
};

struct MoveComponent
{
    float velx, vely, rotation;
};

struct SniffComponent 
{
    float sniffMaxDistance;
};

struct ActivityComponent
{

};