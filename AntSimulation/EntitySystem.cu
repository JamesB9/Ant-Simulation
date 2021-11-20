///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919),
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: EntitySystem.cu
// Description: The system that miniuplates the entities data.
//
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:
//	- (URL)
//
//
//////////////////////////// 80 columns wide //////////////////////////////////
#include "EntitySystem.cuh"

MoveComponent* createMoveComponentArray(int n) {
	MoveComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(MoveComponent));
	return nArray;
}

SniffComponent* createSniffComponentArray(int n) {
	SniffComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(SniffComponent));
	return nArray;
}

ActivityComponent* createActivityComponentArray(int n) {
	ActivityComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(ActivityComponent));
	return nArray;
}

CollisionComponent* createCollisionComponentArray(int n) {
	CollisionComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(CollisionComponent));
	return nArray;
}

__device__ void move(MoveComponent& move, curandState* state, float deltaTime) {
	//Get random vector where {+1 < x > -1, +1 < y > -1}
	Vec2f randomDirection = randomInsideUnitCircle(state);

	//Add randomDirection to the current direction
	move.direction = (move.direction + randomDirection * move.roamStrength);

	//Calculate speed based on direction
	Vec2f targetVelocity = move.direction * move.maxSpeed;

	//Calculate vector to turn to new direction
	Vec2f targetTurningForce = (targetVelocity - move.velocity) * move.turningForce;

	//Clamp new acceleration by maximum turning force
	Vec2f acceleration = clamp(targetTurningForce, move.turningForce);

	//Store current angle
	move.angle = atan2f(move.velocity.y, move.velocity.x);

	//Clamp new velocity to max speed
	move.velocity = clamp(move.velocity + acceleration * deltaTime, move.maxSpeed);
	move.position = move.position + (move.velocity * deltaTime);

	//Debug Output
	//printf("randx %f, randy %f \n", randomDirection.x, randomDirection.y);
	//printf("dirx %.2f, diry %.2f \n", move.direction.x, move.direction.y);
	//printf("tvx %.2f tvy %.2f ttfx %.2f ttfy %.2f \n", targetVelocity.x, targetVelocity.y, targetTurningForce.x, targetTurningForce.y);
	//printf("acx %.2f acy %.2f \n", acceleration.x, acceleration.y);
}


__device__ int getCellIndexDevice(ItemGrid* itemGrid, Map* map, float x, float y) {
	float widthOfCell = 800.0f / itemGrid->worldX;
	float heightOfCell = 800.0f / itemGrid->worldY;
	return (floorf(y / heightOfCell) * itemGrid->worldX) + floorf(x / widthOfCell);
}

__device__ Cell* getCellDevice(ItemGrid* itemGrid, Map* map, float x, float y) {
	return &itemGrid->worldCells[getCellIndexDevice(itemGrid, map, x, y)];
}

__device__ void releasePheromone(ItemGrid* itemGrid, Map* map, MoveComponent& move, ActivityComponent& activity, float deltaTime) {
	activity.timeSinceDrop += deltaTime;

	if (activity.timeSinceDrop > activity.timePerDrop) {
		Cell* cell = getCellDevice(itemGrid, map, move.position.x, move.position.y);
		cell->pheromones[activity.currentActivity] += 0.05f;
		activity.timeSinceDrop = 0;
	}
}

__device__ void sniff(MoveComponent& move, SniffComponent& sniff, float deltaTime) {
}


__device__ void detectWall(MoveComponent& move, CollisionComponent& collision, Map* map, float deltaTime) {
	//Notes for wall detection
	//Cast ray out from and until you hit a 1 in the map
	//if distance from wall to ant is small enough
	//	1. Get the angle between the ant and the wall (1) OR Screen Border
	//	2. invert that angle based on what side of the wall you are on, find the inverse point (mirrored position) of the ant
	//	3. push that new location to the move function for turning

	/*Vec2f topLeft = {0.0f, 0.0f},
		bottomLeft = { 0.0f, 800.0f },
		topRight = { 800.0f, 0.0f },
		bottomRight = { 800.0f, 800.0f };
	Boundary lboundary = { topLeft, bottomLeft, 1 };
	Boundary rboundary = { bottomRight, topRight, 2 };
	Boundary tboundary = { topRight, topLeft, 3 };
	Boundary bboundary = { bottomLeft, bottomRight, 4 };
	Boundary boundaries[4] = { lboundary , rboundary , tboundary , bboundary };
	*/
	//ray position - move.position
	//ray's angle
	Vec2f angle = { cos(move.angle),  sin(move.angle) };
	float targetDistance = 1000000;
	int wallIndex = -1;
	//printf("at angle: %.2f, %.2f\n", angle.x, angle.y);

	for (int i = 0; i < map->wallCount; i++) {
		Boundary& wall = map->walls[i];
		const float x1 = wall.p1.x;
		const float y1 = wall.p1.y;
		const float x2 = wall.p2.x;
		const float y2 = wall.p2.y;

		const float x3 = move.position.x;
		const float y3 = move.position.y;
		const float x4 = move.position.x + (angle.x * 1000.0f);
		const float y4 = move.position.y + (angle.y * 1000.0f);

		const float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
		if (den == 0) { continue; }
		const float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
		const float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;

		if (t > 0 && t < 1 && u > 0) {
			Vec2f targetPosition = { x1 + t * (x2 - x1) , y1 + t * (y2 - y1) };
			float distance = sqrtf(powf(targetPosition.x - move.position.x, 2.0f) + powf(targetPosition.y - move.position.y, 2.0f));

			if (distance < targetDistance) {//Calculate inverse angle
				wallIndex = i;
				targetDistance = distance;
				collision.targetPosition = targetPosition;
			}
		}
	}
	if (wallIndex != -1 && targetDistance < collision.collisionDistance) {
		//Calculate reflected angle
		Vec2f n = clamp(normaliseSurface(map->walls[wallIndex].p1, map->walls[wallIndex].p2), 1.0f);
		Vec2f u = n * (move.velocity.dotProduct(n) / n.dotProduct(n));
		Vec2f w = move.velocity - u;
		//Set reflected angle
		collision.refractionPosition = collision.targetPosition + (clamp(u-w, 1.0f) * targetDistance);
		//Set direction
		if (targetDistance < collision.collisionDistance) {
			move.direction = (u - w);
		}
	}
	else if (wallIndex == -1) {
		move.position = { 400.0f, 400.0f };
	}
	//else {
	//	collision.refractionPosition = collision.targetPosition;
	//}
}

__global__ void simulateEntities(
	Entities* entities,
	float deltaTime,
	ItemGrid* itemGrid,
	Map* map)
{
	//--RNG--
	curandState state;
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(clock(), id, 0, &state);

	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entities->entityCount; i += stride) { // For Each entity for this thread
		move(entities->moves[i], &state ,deltaTime);
		releasePheromone(itemGrid, map, entities->moves[i], entities->activities[i], deltaTime);
		detectWall(entities->moves[i], entities->collisions[i], map, deltaTime);
	}
}

int simulateEntitiesOnGPU(Entities* entities, ItemGrid* itemGrid, Map* map, float deltaTime) {
	// Time Per Drop
	//ActivityComponent::timeSinceDrop +=

	int blockSize = 256;
	int numBlocks = (entities->entityCount + blockSize - 1) / blockSize;

	simulateEntities << <numBlocks, blockSize >> > (
		entities,
		deltaTime,
		itemGrid,
		map);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//std::cout << entities.moves[0].position.x << ", " << entities.moves[0].position.y << std::endl;

	return 0;
}

Entities* initEntities(int entityCount) {
	Entities* entities;
	cudaMallocManaged(&entities, sizeof(Entities));
	entities->entityCount = entityCount;

	entities->moves = createMoveComponentArray(entities->entityCount);
	entities->sniffs = createSniffComponentArray(entities->entityCount);
	entities->activities = createActivityComponentArray(entities->entityCount);
	entities->collisions = createCollisionComponentArray(entities->entityCount);

	for (unsigned int i = 0; i < entities->entityCount; i++) {
		entities->sniffs[i].sniffMaxDistance = 5;

		entities->moves[i].position = { 400.0f, 400.0f };
		entities->moves[i].direction = 0.0f;
		entities->moves[i].velocity = { 0.0f, 0.0f };
		entities->moves[i].maxSpeed = 15.0f;
		entities->moves[i].turningForce = entities->moves[i].maxSpeed * 30.0f;
		entities->moves[i].roamStrength = 2.5f;//2.5f;

		entities->collisions[i].avoid = false;
		entities->collisions[i].targetPosition = {0.0f, 0.0f};
		entities->collisions[i].refractionPosition = { 0.0f, 0.0f };
		entities->collisions[i].collisionDistance = 25.0f;

		entities->activities[i].currentActivity = LEAVING_HOME;
		entities->activities[i].timeSinceDrop = 0.0f;
		entities->activities[i].timePerDrop = 0.1f;
	}

	return entities;
}
/*
int main() {
	Entities entities;
	initEntities(entities);
	printf("%f\n", entities.positions[0].x);
	simulateEntities(entities);
	printf("%f\n", entities.positions[0].x);
	return 0;
}*/
