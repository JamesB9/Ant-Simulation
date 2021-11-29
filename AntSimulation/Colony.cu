#include "Colony.cuh"


Colony* createColoniesArray(int colonyCount) {
	Colony* colonies;
	cudaMallocManaged(&colonies, colonyCount * sizeof(Colony));
	return colonies;
}
