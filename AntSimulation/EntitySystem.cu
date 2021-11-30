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

__device__ void move(MoveComponent& move, float deltaTime) {
	//Get random vector where {+1 < x > -1, +1 < y > -1}
	Vec2f randomDirection = randomInsideUnitCircle(&move.state);

	//Add randomDirection to the current direction
	move.direction = (move.direction + randomDirection * move.roamStrength);

	//Calculate speed based on direction
	Vec2f targetVelocity = move.direction * move.maxSpeed;

	//Calculate vector to turn to new direction
	Vec2f targetTurningForce = (targetVelocity - move.velocity) * move.turningForce;

	//Clamp new acceleration by maximum turning force
	Vec2f acceleration = clamp(targetTurningForce, move.turningForce);

	//Clamp new velocity to max speed
	move.velocity = clamp(move.velocity + acceleration * deltaTime, move.maxSpeed);
	move.position = move.position + (move.velocity * deltaTime);

	//Store current angle
	move.angle = atan2f(move.velocity.y, move.velocity.x);

	//Debug Output
	//printf("randx %f, randy %f \n", randomDirection.x, randomDirection.y);
	//printf("dirx %.2f, diry %.2f \n", move.direction.x, move.direction.y);
	//printf("tvx %.2f tvy %.2f ttfx %.2f ttfy %.2f \n", targetVelocity.x, targetVelocity.y, targetTurningForce.x, targetTurningForce.y);
	//printf("acx %.2f acy %.2f \n", acceleration.x, acceleration.y);
}

__device__ Vec2f getCellCoordinate(ItemGrid* itemGrid, float x, float y) {
	return { floorf(x / itemGrid->cellWidth), floorf(y / itemGrid->cellHeight) };
}

__device__ int getCellIndexDevice(ItemGrid* itemGrid, float x, float y) {
	return (floorf(y / itemGrid->cellHeight) * itemGrid->sizeX) + floorf(x / itemGrid->cellWidth);
}

__device__ Cell* getCellDevice(ItemGrid* itemGrid, float x, float y) {
	return &itemGrid->worldCells[getCellIndexDevice(itemGrid, x, y)];
}

__device__ int getCellIndex(ItemGrid* itemGrid, int x, int y) {
	return (floorf(y) * itemGrid->sizeX) + floorf(x);
}


__device__ void releasePheromone(ItemGrid* itemGrid, MoveComponent& move, ActivityComponent& activity, float deltaTime) {
	activity.timeSinceDrop += deltaTime;

	if (activity.timeSinceDrop > activity.timePerDrop && activity.dropStrength > 0.0f) {
		Cell* cell = getCellDevice(itemGrid, move.position.x, move.position.y);
		cell->pheromones[activity.currentActivity] += activity.dropStrength;
		activity.timeSinceDrop = 0;
	}
	activity.dropStrength -= activity.dropStrengthReduction * deltaTime;
}


__device__ float getPheromoneIntensitySample(ItemGrid* itemGrid, Vec2f position, int sampleRadius, int pheromoneType) {
	Vec2f cellCoordinate = getCellCoordinate(itemGrid, position.x, position.y);
	float totalIntensity = 0;
	for (int dx = cellCoordinate.x - sampleRadius; dx < cellCoordinate.x + sampleRadius; dx++) {
		for (int dy = cellCoordinate.y - sampleRadius; dy < cellCoordinate.y + sampleRadius; dy++) {
			totalIntensity += itemGrid->worldCells[getCellIndex(itemGrid, dx, dy)].pheromones[pheromoneType];
			if (pheromoneType == 1) {
				totalIntensity += itemGrid->worldCells[getCellIndex(itemGrid, dx, dy)].pheromones[2];//Food
			}
		}
	}

	return totalIntensity;
}

__device__ void senseHome(ItemGrid* itemGrid, Colony* colonies, MoveComponent& move, SniffComponent& sniff, ActivityComponent& activity, float deltaTime) {
	Vec2f home = { colonies[activity.colonyId].nestPositionX, colonies[activity.colonyId].nestPositionY };
	float distanceFromHome = getDistance(move.position, home);
	Vec2f vectorToHome;
	vectorToHome = home - move.position;
	vectorToHome = clamp(vectorToHome, 1.0f);

	float distanceFromRememberedFood = getDistance(move.position, activity.lastFoodPickup);
	Vec2f vectorToRememberedFood;
	vectorToRememberedFood = activity.lastFoodPickup - move.position;
	vectorToRememberedFood = clamp(vectorToRememberedFood, 1.0f);

	if (activity.currentActivity == 1) { move.direction = move.direction + (vectorToHome / (distanceFromHome/400.0f)); };
	if (activity.currentActivity == 0 && activity.lastFoodPickup.x != 0.0f && activity.lastFoodPickup.y != 0.0f) { move.direction = move.direction + (vectorToRememberedFood / (distanceFromRememberedFood / 400.0f)); };
}

__device__ void sniff(ItemGrid* itemGrid, Colony* colonies, MoveComponent& move, SniffComponent& sniff, ActivityComponent& activity, float deltaTime) {
	float distance = 15.0f;
	int sampleRadius = 5;
	// Get CELLS
	Cell* currentCell = getCellDevice(itemGrid, move.position.x, move.position.y);
	float baseAngle = atan2f(move.velocity.y, move.velocity.x) + (-90.0f * M_PI / 180);
	//printf("BA: %.2f\n", baseAngle);
	Vec2f leftVector;
	leftVector.x = (-distance * sin(baseAngle - M_PI_4));
	leftVector.y = (distance * cos(baseAngle - M_PI_4));
	
	float leftIntensity = getPheromoneIntensitySample(itemGrid, 
		leftVector + move.position,
		sampleRadius, 
		sniff.sniffPheromone);

	Vec2f rightVector;
	rightVector.x = (-distance * sin(baseAngle + M_PI_4));
	rightVector.y = (distance * cos(baseAngle + M_PI_4));
	
	float rightIntensity = getPheromoneIntensitySample(itemGrid,
		rightVector + move.position,
		sampleRadius, 
		sniff.sniffPheromone);

	Vec2f straightVector;
	straightVector.x = -distance * sin(baseAngle);
	straightVector.y = distance * cos(baseAngle);
	
	float straightIntensity = getPheromoneIntensitySample(itemGrid,
		straightVector + move.position,
		sampleRadius,
		sniff.sniffPheromone);

	float total_intensity = (straightIntensity) + (leftIntensity) + (rightIntensity);
	straightVector = straightVector * (straightIntensity / total_intensity);
	rightVector = rightVector * (rightIntensity / total_intensity);
	leftVector = leftVector * (leftIntensity / total_intensity);
	
	if (total_intensity > 0.0f) {
		//printf("%.2f - %.2f - %.2f\n", (leftIntensity / total_intensity), (straightIntensity / total_intensity), (rightIntensity / total_intensity));
		Vec2f finalVector = rightVector + leftVector + straightVector;
		//printf("%.2f, %.2f\n", finalVector.x, finalVector.y);
		move.direction = finalVector;
	}
	
	/*if (straightIntensity > leftIntensity && straightIntensity > rightIntensity) {
		move.direction = straightVector - move.position;
		//printf("Going straight!\n");
	}
	else if (leftIntensity > rightIntensity) {
		move.direction = leftVector - move.position;
		//printf("Turning left!\n");
	}
	else if (rightIntensity > leftIntensity) {
		move.direction = rightVector - move.position;
		//printf("Turning right!\n");
	}*/

	//printf("IL, IR: %.2f, %.2f\n", leftIntensity, rightIntensity);

	if (activity.currentActivity == 0 && currentCell->foodCount > 0.0f) { // FOOD FOUND!!
		currentCell->foodCount -= 1;
		activity.currentActivity = 1;
		sniff.sniffPheromone = 0;
		activity.dropStrength = activity.maxDropStrength;
		move.direction = { -move.direction.x, -move.direction.y };
		move.velocity = { 0,0 };
		//move.direction = { -move.direction.x, -move.direction.y };
		activity.lastFoodPickup = { move.position.x, move.position.y };
	}
	else if (activity.currentActivity == 1 && currentCell->foodCount > 0.0f) {
		activity.dropStrength = activity.maxDropStrength;
		activity.lastFoodPickup = { move.position.x, move.position.y };
	}

	float nestX = colonies[activity.colonyId].nestPositionX;
	float nestY = colonies[activity.colonyId].nestPositionY;
	float nestRadius = colonies[activity.colonyId].nestRadius;

	if (move.position.x > nestX - nestRadius && move.position.x < nestX + nestRadius && 
		move.position.y > nestY - nestRadius && move.position.y < nestY + nestRadius) { // HOME FOUND!!
		if (activity.currentActivity == 1) {
			move.direction = { -move.direction.x, -move.direction.y };
			move.velocity = { 0,0 };
		}
		activity.currentActivity = 0;
		sniff.sniffPheromone = 1;
		activity.dropStrength = activity.maxDropStrength;
		colonies->totalFood += 1;
	}

}

__device__ void detectWall(MoveComponent& move, CollisionComponent& collision, ActivityComponent& activity, Map* map, float deltaTime) {
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
			//activity.dropStrength *= 0.5f * deltaTime;
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
	Map* map,
	Colony* colonies)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entities->entityCount; i += stride) { // For Each entity for this thread
		move(entities->moves[i], deltaTime);
		releasePheromone(itemGrid, entities->moves[i],  entities->activities[i],  deltaTime);
		sniff(itemGrid, colonies, entities->moves[i], entities->sniffs[i], entities->activities[i], deltaTime);
		senseHome(itemGrid, colonies, entities->moves[i], entities->sniffs[i], entities->activities[i], deltaTime);
		detectWall(entities->moves[i], entities->collisions[i], entities->activities[i], map, deltaTime);
	}
}

int simulateEntitiesOnGPU(Entities* entities, ItemGrid* itemGrid, Map* map, Colony* colonies, float deltaTime) {
	// Time Per Drop
	//ActivityComponent::timeSinceDrop +=

	int blockSize = 256;
	int numBlocks = (entities->entityCount + blockSize - 1) / blockSize;

	simulateEntities << <numBlocks, blockSize >> > (
		entities,
		deltaTime,
		itemGrid,
		map,
		colonies);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return 0;
}


__global__ void setupStates(Entities* entities) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entities->entityCount; i += stride) { // For Each entity for this thread
		//--RNG--
		curand_init(clock(), index, 0, &entities->moves[i].state);
	}
}

void setupStatesOnGPU(Entities* entities) {

	int blockSize = 256;
	int numBlocks = (entities->entityCount + blockSize - 1) / blockSize;

	setupStates << <numBlocks, blockSize >> > (entities);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
}

Entities* initEntities(Colony* colonies, int entityCount) {
	Entities* entities;
	cudaMallocManaged(&entities, sizeof(Entities));
	entities->entityCount = entityCount;

	entities->moves = createMoveComponentArray(entities->entityCount);
	entities->sniffs = createSniffComponentArray(entities->entityCount);
	entities->activities = createActivityComponentArray(entities->entityCount);
	entities->collisions = createCollisionComponentArray(entities->entityCount);

	for (unsigned int i = 0; i < entities->entityCount; i++) {
		
		entities->activities[i].colonyId = 0; // CHANGE LATER

		entities->sniffs[i].sniffMaxDistance = Config::ANT_MAX_SNIFF_DISTANCE;
		entities->sniffs[i].sniffPheromone = FOUND_FOOD;

		entities->moves[i].position = { 
			colonies[entities->activities[i].colonyId].nestPositionX, 
			colonies[entities->activities[i].colonyId].nestPositionY 
		};
		entities->moves[i].direction = { 0.0f, 0.0f };
		entities->moves[i].velocity = { 0.0f, 0.0f };
		entities->moves[i].maxSpeed = Config::ANT_MAX_SPEED;
		entities->moves[i].turningForce = Config::ANT_TURN_FORCE;
		entities->moves[i].roamStrength = Config::ANT_ROAM_STRENGTH;

		entities->collisions[i].avoid = false;
		entities->collisions[i].targetPosition = {0.0f, 0.0f};
		entities->collisions[i].refractionPosition = { 0.0f, 0.0f };
		entities->collisions[i].collisionDistance = Config::ANT_COLLISION_DISTANCE;

		entities->activities[i].currentActivity = LEAVING_HOME;
		entities->activities[i].dropStrength = Config::INITIAL_DROP_STRENGTH;
		entities->activities[i].dropStrengthReduction = Config::DROP_STRENGTH_REDUCTION;
		entities->activities[i].timeSinceDrop = 0.0f;
		entities->activities[i].timePerDrop = Config::PHEROMONE_DROP_TIME;
		entities->activities[i].maxDropStrength = Config::INITIAL_DROP_STRENGTH;
		entities->activities[i].lastFoodPickup = { 0.0f, 0.0f };
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
