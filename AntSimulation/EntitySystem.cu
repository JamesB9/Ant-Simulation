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

////////////////////////////////////////////////////////////
/// \brief Get a cells coordinate based from location within the render window
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param x: Screen coordinate x
/// \param y: Screen coordinate y
///
///
////////////////////////////////////////////////////////////
__device__ Vec2f getCellCoordinate(ItemGrid* itemGrid, float x, float y) {
	return { floorf(x / itemGrid->cellWidth), floorf(y / itemGrid->cellHeight) };
}

////////////////////////////////////////////////////////////
/// \brief Get a cell's index based on location within the render window
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param x: Screen coordinate x
/// \param y: Screen coordinate y
///
///
////////////////////////////////////////////////////////////
__device__ int getCellIndexDevice(ItemGrid* itemGrid, float x, float y) {
	return (floorf(y / itemGrid->cellHeight) * itemGrid->sizeX) + floorf(x / itemGrid->cellWidth);
}

////////////////////////////////////////////////////////////
/// \brief Get a cell based on a location within the render window
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param x: Screen coordinate x
/// \param y: Screen coordinate y
///
///
////////////////////////////////////////////////////////////
__device__ Cell* getCellDevice(ItemGrid* itemGrid, float x, float y) {
	return &itemGrid->worldCells[getCellIndexDevice(itemGrid, x, y)];
}


////////////////////////////////////////////////////////////
/// \brief Get a cell's index based on location within the render window
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param x: Screen coordinate x
/// \param y: Screen coordinate y
///
///
////////////////////////////////////////////////////////////
__device__ int getCellIndex(ItemGrid* itemGrid, int x, int y) {
	return (floorf(y) * itemGrid->sizeX) + floorf(x);
}

////////////////////////////////////////////////////////////
/// \brief Release pheromone function to add to a world cell
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param move: Move Component for an ant
/// \param activity: Activity Component for an ant
/// \param deltaTime: Time taken in ms between frame rendering
///
///
////////////////////////////////////////////////////////////
__device__ void releasePheromone(ItemGrid* itemGrid, MoveComponent& move, ActivityComponent& activity, CollisionComponent& collision, float deltaTime) {
	if (collision.stopPheromone) { return; }
	activity.timeSinceDrop += deltaTime;

	if (activity.timeSinceDrop > activity.timePerDrop && activity.dropStrength > 0.0f) {
		Cell* cell = getCellDevice(itemGrid, move.position.x, move.position.y);
		cell->pheromones[activity.currentActivity] += activity.dropStrength;
		activity.timeSinceDrop = 0;
	}
	activity.dropStrength -= activity.dropStrengthReduction * deltaTime;
}

////////////////////////////////////////////////////////////
/// \brief Get the total amount of a certain pheromone at a position within a radius
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param position: Vector2f of the position to sample
/// \param sampleRadius: Radius from the point to take a sample from
/// \param pheromoneType: Chosen type of pheromone to count
///
///
////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////
/// \brief Alter an ant's direction more towards home the closer they are
///
/// Ants have an inbuilt extra sensory perspective whereby they know the general direction to their colony
/// this function mimics that same sensory advantage, the closer the ant is to home, the easier they move towards it.
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param colonies: Pointer to a list of colonies
/// \param move: Move Component for an ant
/// \param sniff: Sniff Component for an ant
/// \param activity: Activity Component for an ant
/// \param deltaTime: Time taken in ms between frame rendering
///
////////////////////////////////////////////////////////////
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

	if (activity.currentActivity == 1) { move.direction = move.direction + (vectorToHome * distanceFromRememberedFood/distanceFromHome); };
	//if (activity.currentActivity == 0 && activity.lastFoodPickup.x != 0.0f && activity.lastFoodPickup.y != 0.0f) { move.direction = move.direction + (vectorToRememberedFood * distanceFromRememberedFood/distanceFromHome); };
}

////////////////////////////////////////////////////////////
/// \brief Sniff function to help ants navigate around the map
///
/// Using 3 points infront of the ant; left, middle and right
/// the pheromone for these 3 areas is taken then divided by the total
/// from said 3 areas to provide a stregth. Each direction is multiplied by the strength,
/// the ant's direction is when set as the addition of these 3 vectors
///
/// \param itemGrid: World's designated ItemGrid passed as a pointer
/// \param colonies: Pointer to a list of colonies
/// \param move: Move Component for an ant
/// \param sniff: Sniff Component for an ant
/// \param activity: Activity Component for an ant
/// \param deltaTime: Time taken in ms between frame rendering
///
////////////////////////////////////////////////////////////
__device__ void sniff(ItemGrid* itemGrid, Colony* colonies, MoveComponent& move, SniffComponent& sniff, ActivityComponent& activity, float deltaTime) {

	//Sample distance from ant
	float distance = 15.0f;
	//Sample radius from the 3 points
	int sampleRadius = 5;


	// Get the Ant's current cell
	Cell* currentCell = getCellDevice(itemGrid, move.position.x, move.position.y);

	//Base angle of the ant in radians
	float baseAngle = atan2f(move.velocity.y, move.velocity.x) + (-90.0f * M_PI / 180);

	//Left vector calculated using the distance and base angle - 45 degrees
	Vec2f leftVector;
	leftVector.x = (-distance * sin(baseAngle - M_PI_4));
	leftVector.y = (distance * cos(baseAngle - M_PI_4));

	//The total count of the target pheromone found within the calculated position
	float leftIntensity = getPheromoneIntensitySample(itemGrid,
		leftVector + move.position,
		sampleRadius,
		sniff.sniffPheromone);

	//Right vector calculated using the distance and base angle + 45 degrees
	Vec2f rightVector;
	rightVector.x = (-distance * sin(baseAngle + M_PI_4));
	rightVector.y = (distance * cos(baseAngle + M_PI_4));

	//The total count of the target pheromone found within the calculated position
	float rightIntensity = getPheromoneIntensitySample(itemGrid,
		rightVector + move.position,
		sampleRadius,
		sniff.sniffPheromone);

	//Right vector calculated using the distance and base angle
	Vec2f straightVector;
	straightVector.x = -distance * sin(baseAngle);
	straightVector.y = distance * cos(baseAngle);

	//The total count of the target pheromone found within the calculated position
	float straightIntensity = getPheromoneIntensitySample(itemGrid,
		straightVector + move.position,
		sampleRadius,
		sniff.sniffPheromone);

	//Sum of the 3 intensity readings
	float total_intensity = (straightIntensity) + (leftIntensity) + (rightIntensity);

	//Each directional vector from the ant is multiplied by its intensity / total_intensity
	straightVector = straightVector * (straightIntensity / total_intensity);
	rightVector = rightVector * (rightIntensity / total_intensity);
	leftVector = leftVector * (leftIntensity / total_intensity);

	if (total_intensity > 0.0f) {
		//printf("%.2f - %.2f - %.2f\n", (leftIntensity / total_intensity), (straightIntensity / total_intensity), (rightIntensity / total_intensity));
		Vec2f finalVector = rightVector + leftVector + straightVector;
		//printf("%.2f, %.2f\n", finalVector.x, finalVector.y);
		move.direction = finalVector;
	}

	//If the current activity == 0 (Looking for food) and the current cell we're on has food
	if (activity.currentActivity == 0 && currentCell->foodCount > 0.0f) {
		//Take "food" from cell
		currentCell->foodCount -= 1;
		//Current activity = 1 (Return to colony)
		activity.currentActivity = 1;
		//Pheromone the ant will sniff for == 0 (Leaving home pheromone)
		sniff.sniffPheromone = 0;
		//Strength of the ant's pheromone is reset (highest)
		activity.dropStrength = activity.maxDropStrength;
		//Set the ant direction to the reverse of its current
		move.direction = { -move.direction.x, -move.direction.y };
		//Stop the ant
		move.velocity = { 0,0 };
		//Set the position for its lastFoodPickup (see senseHome)
		activity.lastFoodPickup = { move.position.x, move.position.y };
	}
	//If current activity == 1 (Looking for colony) and the current cell we're on has food
	else if (activity.currentActivity == 1 && currentCell->foodCount > 0.0f) {
		//Strength of the ant's pheromone is reset (highest)
		activity.dropStrength = activity.maxDropStrength;
		//Set the position for its lastFoodPickup (see senseHome)
		activity.lastFoodPickup = { move.position.x, move.position.y };
	}

	//Get the current ant's colony via its stored colony ID
	float nestX = colonies[activity.colonyId].nestPositionX;
	float nestY = colonies[activity.colonyId].nestPositionY;
	float nestRadius = colonies[activity.colonyId].nestRadius;

	//If the ant is within the radius of the nest/colony
	if (move.position.x > nestX - nestRadius && move.position.x < nestX + nestRadius &&
		move.position.y > nestY - nestRadius && move.position.y < nestY + nestRadius) {
		//If current activity == 1 (Looking for colony)
		if (activity.currentActivity == 1) {
			//Set the ant direction to the reverse of its current
			move.direction = { -move.direction.x, -move.direction.y };
			//Stop the ant
			move.velocity = { 0,0 };
			//Current activity = 0 (Look for food)
			activity.currentActivity = 0;
			//Pheromone the ant will sniff for == 1 (Found food pheromone)
			sniff.sniffPheromone = 1;
			//Add food to colony
			colonies->totalFood += 1;
		}
		//Strength of the ant's pheromone is reset (highest)
		activity.dropStrength = activity.maxDropStrength;
	}

}

////////////////////////////////////////////////////////////
/// \brief Detect walls using a ray cast from the ant
///
/// Boundaries/Walls are defined as a line made from two points,
/// because we use vertex arrays for the storage of our world map
/// it makes sense to take advantage of ray casting.
/// When ray casting we can also find the inverse position and angle
/// based around the point of incidence.
///
/// \param move: Move Component for an ant
/// \param collision: Collision Component for an ant
/// \param activity: Activity Component for an ant
/// \param map: (Custom Class Map) containing vertecies of the world
/// \param deltaTime: Time taken in ms between frame rendering
///
////////////////////////////////////////////////////////////
__device__ void detectWall(MoveComponent& move, CollisionComponent& collision, ActivityComponent& activity, Map* map, float deltaTime, Colony* colonies) {


	//Angle of the ray == ant's current direction
	Vec2f angle = { cos(move.angle),  sin(move.angle) };
	//Initial target distance <-- unachieveable distance (to be whittled down)
	float targetDistance = 1000000;
	//Initial index of the closest target wall (-1 == no index found so far)
	int wallIndex = -1;

	//For each boundary/wall in the map
	for (int i = 0; i < map->wallCount; i++) {

		/*
			Line-Line intersection between (Ray from ant to essentially infinity) and (Two points defining a boundary)
			https://www.wikiwand.com/en/Line�line_intersection
		*/

		Boundary& wall = map->walls[i];
		const float x1 = wall.p1.x;
		const float y1 = wall.p1.y;
		const float x2 = wall.p2.x;
		const float y2 = wall.p2.y;

		const float x3 = move.position.x;
		const float y3 = move.position.y;
		const float x4 = move.position.x + (angle.x * 1000.0f); //Angle is multiplied here so it can reach a boundary at a large distance
		const float y4 = move.position.y + (angle.y * 1000.0f);

		const float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
		if (den == 0) { continue; }
		const float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
		const float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den;

		if (t > 0 && t < 1 && u > 0) {
			//Point at which the lines intersect
			Vec2f targetPosition = { x1 + t * (x2 - x1) , y1 + t * (y2 - y1) };
			//Simple pythagorus theorem
			float distance = sqrtf(powf(targetPosition.x - move.position.x, 2.0f) + powf(targetPosition.y - move.position.y, 2.0f));

			//If the distance from the bounday is less than the current targetDistance (initially set to unachieveable number)
			if (distance < targetDistance) {
				//Store the current wall in memory
				wallIndex = i;
				targetDistance = distance;
				collision.targetPosition = targetPosition;
				collision.stopPheromone = true;
			}
			else {
				collision.stopPheromone = false;
			}
		}
	}
	//If the wall we have stored in memory is valid and its distance is less than
	//or equal too the currently set collision distance for ants
	if (wallIndex != -1 && targetDistance <= collision.collisionDistance) {

		/*
			Calculation of reflected angle (Bounce Angle)
			https://stackoverflow.com/a/573206/16096878
			^ Mathematical implementation only
		*/

		//Surface normal n
		Vec2f n = clamp(normaliseSurface(map->walls[wallIndex].p1, map->walls[wallIndex].p2), 1.0f);

		//Vector perpendicular to the wall from the position of the ant
		Vec2f u = n * (move.velocity.dotProduct(n) / n.dotProduct(n));

		//Vector parallel to the wall
		Vec2f w = move.velocity - u;

		//Refraction position = Line-Line point of intersection + u-w multiplied
		//by the distance from the ant to the point of the Line-Line intersection
		collision.refractionPosition = collision.targetPosition + (clamp(u-w, 1.0f) * targetDistance);

		//Change the move direction to the "bounce angle"
		move.direction = (u - w);
	}
	//If the wall index == -1 then the ant is looking at nothing (which is (technically) impossible)
	//however, ants sometimes find themselves outside the bounds of the map
	else if (wallIndex == -1) {
		move.position = { colonies[activity.colonyId].nestPositionX, colonies[activity.colonyId].nestPositionY };
	}
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
		releasePheromone(itemGrid, entities->moves[i], entities->activities[i], entities->collisions[i], deltaTime);
		sniff(itemGrid, colonies, entities->moves[i], entities->sniffs[i], entities->activities[i], deltaTime); // Try to Optimise
		senseHome(itemGrid, colonies, entities->moves[i], entities->sniffs[i], entities->activities[i], deltaTime);
		detectWall(entities->moves[i], entities->collisions[i], entities->activities[i], map, deltaTime, colonies); // Try to Optimise
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
		entities->collisions[i].stopPheromone = false;

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
