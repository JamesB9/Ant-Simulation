
#define LEAVING_HOME 0;
#define FOUND_FOOD 1;

struct MoveComponent
{
    float x, y;
    float speed, rotation;
};

struct SniffComponent 
{
    float sniffMaxDistance;
};

struct ActivityComponent
{

};