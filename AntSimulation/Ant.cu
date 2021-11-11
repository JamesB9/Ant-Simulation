#include "ant.cuh"


PositionComponent* createPositionComponentArray(int n) {
	PositionComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(PositionComponent));
	return nArray;
}

MoveComponent* createMoveComponentArray(int n) {
	MoveComponent* nArray;
	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&nArray, n * sizeof(MoveComponent));
	return nArray;
}

__device__ void move(PositionComponent& position, MoveComponent& move, float deltaTime) {
	// Calculate Velocity
	float vx = deltaTime * move.velx * sin(move.rotation);
	float vy = deltaTime * move.vely * cos(move.rotation);
	position.x += vx;
	position.y += vy;
}

__global__ void moveSystem(PositionComponent* positions, MoveComponent* moves, int entityCount, float deltaTime) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; // Index of the current thread within its block
	int stride = blockDim.x * gridDim.x; // Number of threads in the block
	for (int i = index; i < entityCount; i += stride) {
		move(positions[i], moves[i], deltaTime);
	}
}

int simulateEntities(Entities& entities, float deltaTime) {
	// Run kernel on 1M elements on the CPU
	int blockSize = 256;
	int numBlocks = (entities.entityCount + blockSize - 1) / blockSize;
	moveSystem << <numBlocks, blockSize >> > (entities.positions, entities.moves, entities.entityCount, deltaTime);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return 0;
}

int initEntities(Entities& entities) {
	entities.positions = createPositionComponentArray(entities.entityCount);
	entities.moves = createMoveComponentArray(entities.entityCount);

	for (unsigned int i = 0; i < entities.entityCount; i++) {
		entities.positions[i].x = 400;
		entities.positions[i].y = 400;
		entities.moves[i].rotation = 3.14159265f*2.0f * i / entities.entityCount;
		entities.moves[i].velx = 10;
		entities.moves[i].vely = 10;
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