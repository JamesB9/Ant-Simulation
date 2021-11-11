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

__device__ void move(MoveComponent& move, float deltaTime) {
	// Calculate Velocity
	float vx = deltaTime * move.speed * sin(move.rotation);
	float vy = deltaTime * move.speed * cos(move.rotation);
	move.x += vx;
	move.y += vy;
}

__device__ void releasePheromone(MoveComponent& move, float deltaTime) {
	// Calculate Velocity
	float vx = deltaTime * move.speed * sin(move.rotation);
	float vy = deltaTime * move.speed * cos(move.rotation);
	move.x += vx;
	move.y += vy;
}

__device__ void sniff(MoveComponent& move, SniffComponent& sniff, float deltaTime) {

}

__global__ void simulateEntities(
	MoveComponent* moves, 
	SniffComponent* sniffs,
	int entityCount, 
	float deltaTime) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entityCount; i += stride) { // For Each entity for this thread
		move(moves[i], deltaTime);
	}
}

int simulateEntitiesOnGPU(Entities& entities, float deltaTime) {
	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (entities.entityCount + blockSize - 1) / blockSize;
	simulateEntities << <numBlocks, blockSize >> > (
		entities.moves, 
		entities.sniffs, 
		entities.entityCount, 
		deltaTime);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return 0;
}

int initEntities(Entities& entities) {
	entities.moves = createMoveComponentArray(entities.entityCount);
	entities.sniffs = createSniffComponentArray(entities.entityCount);

	for (unsigned int i = 0; i < entities.entityCount; i++) {
		entities.moves[i].x = 400;
		entities.moves[i].y = 400;
		entities.moves[i].rotation = 3.14159265f*2.0f * i / entities.entityCount;
		entities.moves[i].speed = 10;
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