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

__device__ void move(MoveComponent& move, float deltaTime) {
	// Calculate Velocity
	float vx = deltaTime * move.speed * sin(move.rotation);
	float vy = deltaTime * move.speed * cos(move.rotation);
	move.x += vx;
	move.y += vy;
}

__device__ int getCellIndexDevice2(ItemGrid* itemGrid, float x, float y) {
	return (floorf(y) * itemGrid->worldX) + floorf(x);
}

__device__ Cell* getCellDevice2(ItemGrid* itemGrid, float x, float y) {
	return &itemGrid->worldCells[getCellIndexDevice2(itemGrid, x, y)];
}

__device__ void releasePheromone(ItemGrid* itemGrid, MoveComponent& move, ActivityComponent& activity) {
	Cell* cell = getCellDevice2(itemGrid, move.x, move.y);
	cell->pheromones[activity.currentActivity] += 0.5f;
}

__device__ void sniff(MoveComponent& move, SniffComponent& sniff, float deltaTime) {

}

__global__ void simulateEntities(
	MoveComponent* moves, 
	SniffComponent* sniffs,
	ActivityComponent* activities,
	int entityCount, 
	float deltaTime,
	ItemGrid* itemGrid) 
{

	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entityCount; i += stride) { // For Each entity for this thread
		move(moves[i], deltaTime);
		releasePheromone(itemGrid, moves[i], activities[i]);
	}
}

int simulateEntitiesOnGPU(Entities& entities, ItemGrid* itemGrid, float deltaTime) {
	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (entities.entityCount + blockSize - 1) / blockSize;
	
	simulateEntities << <numBlocks, blockSize >> > (
		entities.moves, 
		entities.sniffs, 
		entities.activities,
		entities.entityCount, 
		deltaTime,
		itemGrid);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return 0;
}

int initEntities(Entities& entities) {
	entities.moves = createMoveComponentArray(entities.entityCount);
	entities.sniffs = createSniffComponentArray(entities.entityCount);
	entities.activities = createActivityComponentArray(entities.entityCount);

	for (unsigned int i = 0; i < entities.entityCount; i++) {
		entities.moves[i].x = 400;
		entities.moves[i].y = 400;
		entities.moves[i].rotation = 2 * M_PI * i / entities.entityCount;
		entities.moves[i].speed = 50;

		entities.sniffs[i].sniffMaxDistance = 5;
		entities.activities[i].currentActivity = LEAVING_HOME;
	}

	return 0;
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