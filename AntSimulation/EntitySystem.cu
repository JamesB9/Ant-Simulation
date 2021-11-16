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
#include "Utilities.cuh"


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
	move.direction += randomInsideUnitCircle(move.position.y) * 0.125f;

	//printf("dirx %.2f, diry %.2f \n", move.direction.x, move.direction.y);

	Vec2f targetVelocity = move.direction * move.maxSpeed;
	Vec2f targetTurningForce = (targetVelocity - move.velocity) * move.turningForce;

	//printf("tvx %.2f tvy %.2f ttfx %.2f ttfy %.2f \n", targetVelocity.x, targetVelocity.y, targetTurningForce.x, targetTurningForce.y);

	Vec2f acceleration = clamp(targetTurningForce, move.turningForce);

	//printf("acx %.2f acy %.2f \n", acceleration.x, acceleration.y);

	move.velocity = clamp(move.velocity + acceleration * deltaTime, move.maxSpeed);
	move.position = move.position + (move.velocity * deltaTime);
	
}

__device__ void releasePheromone(MoveComponent& move, float deltaTime) {
	// Calculate Velocity
	
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

	//std::cout << entities.moves[0].position.x << ", " << entities.moves[0].position.y << std::endl;

	return 0;
}

int initEntities(Entities& entities) {
	entities.moves = createMoveComponentArray(entities.entityCount);
	entities.sniffs = createSniffComponentArray(entities.entityCount);

	for (unsigned int i = 0; i < entities.entityCount; i++) {
		entities.moves[i].position = { 400.0f, 400.0f };
		entities.moves[i].direction = 3.14159265f*2.0f * i / entities.entityCount;
		entities.moves[i].velocity = { 0.0f, 0.0f };
		entities.moves[i].maxSpeed = 25.0f;
		entities.moves[i].turningForce = 25.0f;
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