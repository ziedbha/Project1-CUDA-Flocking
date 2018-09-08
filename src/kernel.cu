#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 16

// Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// Buffers that represent the reshuffled versions of pos/vel1/vel2
glm::vec3 *dev_posCpy;
glm::vec3 *dev_vel1Cpy;
glm::vec3 *dev_vel2Cpy;

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* This is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	// This is basic CUDA memory management and error checking.
	// Don't forget to cudaFree in  Boids::endSimulation.
	cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

	// This is a typical CUDA kernel invocation.
	kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numObjects,
		dev_pos, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// Computing grid params
	gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
	gridSideCount = 2 * halfSideCount;

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x -= halfGridWidth;
	gridMinimum.y -= halfGridWidth;
	gridMinimum.z -= halfGridWidth;

	// Additional buffers for scattered grid
	cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

	cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

	cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

	cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

	// Additional buffers for coherent grid
	cudaMalloc((void**)&dev_posCpy, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_posCpy failed!");

	cudaMalloc((void**)&dev_vel1Cpy, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1Cpy failed!");

	cudaMalloc((void**)&dev_vel2Cpy, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_vel2Cpy failed!");

	// Wrap the key/value buffers around the thrust pointers
	dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);
	dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

	cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_vel1, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 alignment(0.0f);
	glm::vec3 separation(0.0f);
	glm::vec3 cohesion(0.0f);
	glm::vec3 deltaVel(0.0f);

	float alignmentCount = 0.0f;
	float cohesionCount = 0.0f;

	for (int i = 0; i < N; i++) {
		if (i == iSelf) {
			continue;
		}
		glm::vec3 otherPos = pos[i];
		float distance = glm::distance(pos[iSelf], otherPos);

		// Rule 1 - Alignment: boids fly towards their local perceived center of mass, which excludes themselves
		if (distance < rule1Distance) {
			cohesion += otherPos;
			++cohesionCount;
		}

		// Rule 2 - Separation: boids try to stay a distance d away from each other
		if (distance < rule2Distance) {
			separation -= otherPos - pos[iSelf];
		}

		// Rule 3 - Cohesion: boids try to match the speed of surrounding boids
		if (distance < rule3Distance) {
			alignment += vel[i];
			++alignmentCount;
		}
	}

	// Average out the cohesion velocity and scale it
	if (cohesionCount > 0) {
		cohesion /= cohesionCount;
		cohesion = (cohesion - pos[iSelf]) * rule1Scale;
		deltaVel += cohesion;
	}
	

	// Scale the separation velocity
	separation *= rule2Scale;
	deltaVel += separation;

	// Average out the cohesion velocity and scale it
	if (alignmentCount > 0) {
		alignment *= rule3Scale / alignmentCount;
		deltaVel += alignment;
	}

	return deltaVel;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Compute new velocity and clamp it
	glm::vec3 deltaVel = computeVelocityChange(N, index, pos, vel1);
	glm::vec3 newVel = vel1[index] + deltaVel;
	float newSpeed = glm::length(newVel);
	newVel = newSpeed > maxSpeed ? glm::normalize(newVel) * maxSpeed : newVel;

	// Record the new velocity into vel2. Question: why NOT vel1? Answer: because vel1 always contains
	// the velocity of the previous frame update. After updating the current frame with the new velocity (vel2)
	// we set vel1 = vel2, which is the entire purpose of ping-pong velocity
	vel2[index] = newVel;
}

/**
* Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	thisPos += vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

	thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// Consider this method of computing a 1D index from a 3D grid index.
// Since memory is contiguous, it is best to iterate over step by step
// So for z then y then x is better (since x goes 1 index by 1 index)
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution, glm::vec3 gridMin, float inverseCellWidth,
								   glm::vec3 *pos, int *indices, int *gridIndices) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		glm::vec3 offsetPos = pos[index] + glm::vec3(scene_scale); // all boids are now in [0, 2 * scene_scale], makes indexing easier
		offsetPos *= inverseCellWidth; // how many cell width is each vector component
		glm::ivec3 cellIndex(glm::floor(offsetPos));
		gridIndices[index] = gridIndex3Dto1D(cellIndex.x, cellIndex.y, cellIndex.z, gridResolution);

		// ith boid has its data in ith position in pos/vel1/vel2 arrays (trivially, but important since we will sort)
		indices[index] = index;
	}
}

// Consider how this could be useful for indicating that a cell
// does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

/* This kernel is called after sorting is done, which means that a threadId will represent some boid
   and not necessarily the threadIdth boid (because its a permutation)
*/
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices, int *gridCellStartIndices, 
										 int *gridCellEndIndices) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x; // ach index represents some boid
	if (i > 0 && i < N) {
		int cell = particleGridIndices[i]; // The cell of the ith boid
		int prevCell = particleGridIndices[i - 1]; // The cell of the i - 1th boid
		if (prevCell != cell) {
			// If the cells are not the same, then we have a new cell! (and end the previous one)
			gridCellStartIndices[cell] = i;
			gridCellEndIndices[prevCell] = i - 1;
		}
	} else if (i == 0) {
		gridCellStartIndices[particleGridIndices[0]] = 0;
	}
	
	if (i == N - 1) {
		gridCellEndIndices[particleGridIndices[N-1]] = N - 1;
	}
}

/* This determines which octant of the cell the boid is in. E.g if x coord > 0.5f, then the boid is in the right half
   of the cell, and the same logic applies to y and z to determine an octant. An octant is represented by a vec3 
   (direction). I use the vec3 to compute the offset from the original cellIndex to get the other 7 neighbors
*/
__device__ void findCellNeighbors(int* outNeighborGrids, glm::vec3 offsetPos, int gridResolution, int cellIndex, int cellIndexMin, int cellIndexMax) {
	glm::ivec3 direction(1, 1, 1);
	offsetPos -= glm::floor(offsetPos);
	if (offsetPos.x < 0.5f) {
		direction.x = -1;
	}

	if (offsetPos.y < 0.5f) {
		direction.y = -1;
	}

	if (offsetPos.z < 0.5f) {
		direction.z = -1;
	}

	// Neighbors are ordered at a different order (from lowest index to highest index)
	outNeighborGrids[0] = cellIndex;
	outNeighborGrids[1] = cellIndex + direction.x;
	outNeighborGrids[2] = cellIndex + direction.y * gridResolution;
	outNeighborGrids[3] = cellIndex + direction.x + direction.y * gridResolution;
	outNeighborGrids[4] = cellIndex + direction.z * gridResolution * gridResolution;
	outNeighborGrids[5] = cellIndex + direction.x + direction.z * gridResolution * gridResolution;
	outNeighborGrids[6] = cellIndex + direction.y * gridResolution + direction.z * gridResolution * gridResolution;
	outNeighborGrids[7] = cellIndex + direction.x + direction.y * gridResolution + direction.z * gridResolution * gridResolution;

	for (int i = 0; i < 8; i++) {
		if (outNeighborGrids[i] > cellIndexMax || outNeighborGrids[i] < cellIndexMin) {
			outNeighborGrids[i] = -1;
		}
	}
	return;
}

__global__ void kernUpdateVelNeighborSearchScattered(int N, int gridResolution, glm::vec3 gridMin,
													 float inverseCellWidth, float cellWidth,
													 int *gridCellStartIndices, int *gridCellEndIndices,
													 int *particleArrayIndices,
													 glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= N) {
		return;
	}

	glm::vec3 thisPos = pos[particleArrayIndices[index]] + glm::vec3(scene_scale);
	thisPos *= inverseCellWidth;
	glm::ivec3 gridIndex(glm::floor(thisPos));
	int gridCell = gridIndex3Dto1D(gridIndex.x, gridIndex.y, gridIndex.z, gridResolution);

	// Identify neighboring cells
	glm::vec3 gridMax((gridResolution - 1) * cellWidth);
	int maxCell = gridIndex3Dto1D(gridMax.x, gridMax.y, gridMax.z, gridResolution);
	int minCell = 0;
	int neighbors[8];
	findCellNeighbors(neighbors, thisPos, gridResolution, gridCell, minCell, maxCell);

	// Compute delta vel
	glm::vec3 alignment(0.0f);
	glm::vec3 separation(0.0f);
	glm::vec3 cohesion(0.0f);
	glm::vec3 deltaVel(0.0f);
	int alignmentCount = 0;
	int cohesionCount = 0;

	for (int i = 0; i < 8; i++) {
		int neighborIndex = neighbors[i];
		if (neighborIndex != -1) {
			int start = gridCellStartIndices[neighborIndex];
			int end = gridCellEndIndices[neighborIndex];
			if (start != -1 && end != -1) {
				for (int j = start; j <= end; j++) {
					if (j != index) {
						glm::vec3 otherPos = pos[particleArrayIndices[j]];

						float distance = glm::length(pos[particleArrayIndices[index]] - otherPos);

						// Rule 1 - Cohesion: boids fly towards their local perceived center of mass, which excludes themselves
						if (distance < rule1Distance) {
							cohesion += otherPos;
							++cohesionCount;
						}

						// Rule 2 - Separation: boids try to stay a distance d away from each other
						if (distance < rule2Distance) {
							separation -= otherPos - pos[particleArrayIndices[index]];
						}

						// Rule 3 - Alignment: boids try to match the speed of surrounding boids
						if (distance < rule3Distance) {
							alignment += vel1[particleArrayIndices[j]];
							++alignmentCount;
						}
					}
				}
			}
		}
	}

	// Average out the cohesion velocity and scale it
	if (cohesionCount > 0) {
		cohesion /= cohesionCount;
		cohesion = (cohesion - pos[particleArrayIndices[index]]) * rule1Scale;
		deltaVel += cohesion;
	}

	// Scale the separation velocity
	separation *= rule2Scale;
	deltaVel += separation;

	// Average out the cohesion velocity and scale it
	if (alignmentCount > 0) {
		alignment *= rule3Scale / alignmentCount;
		deltaVel += alignment;
	}

	glm::vec3 newVel = vel1[particleArrayIndices[index]] + deltaVel;
	float newSpeed = glm::length(newVel);
	newVel = newSpeed > maxSpeed ? (newVel / newSpeed) * maxSpeed : newVel;
	vel2[particleArrayIndices[index]] = newVel;
}

/* We need 1 additional buffer for each data array to avoid synchornization issues and memory access issues since
   we are overwriting data */
__global__ void kernShuffleDataArrays(int* indices, glm::vec3* pos, glm::vec3* posCpy,
									  glm::vec3* vel1, glm::vec3* vel1Cpy, glm::vec3* vel2, glm::vec3* vel2Cpy) {
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Remap each data entry at index provided by the indices array to index threadId
	posCpy[threadId] = pos[indices[threadId]];
	vel1Cpy[threadId] = vel1[indices[threadId]];
	vel2Cpy[threadId] = vel2[indices[threadId]];
}

__global__ void kernUpdateVelNeighborSearchCoherent(int N, int gridResolution, glm::vec3 gridMin,
													float inverseCellWidth, float cellWidth,
													int *gridCellStartIndices, int *gridCellEndIndices,
													glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= N) {
		return;
	}

	glm::vec3 thisPos = pos[index] + glm::vec3(scene_scale);
	thisPos *= inverseCellWidth;
	glm::ivec3 gridIndex(glm::floor(thisPos));
	int gridCell = gridIndex3Dto1D(gridIndex.x, gridIndex.y, gridIndex.z, gridResolution);

	// Identify neighboring cells
	glm::vec3 gridMax((gridResolution - 1) * cellWidth);
	int maxCell = gridIndex3Dto1D(gridMax.x, gridMax.y, gridMax.z, gridResolution);
	int minCell = 0;
	int neighbors[8];
	findCellNeighbors(neighbors, thisPos, gridResolution, gridCell, minCell, maxCell);

	// Compute delta vel
	glm::vec3 alignment(0.0f);
	glm::vec3 separation(0.0f);
	glm::vec3 cohesion(0.0f);
	glm::vec3 deltaVel(0.0f);
	int alignmentCount = 0;
	int cohesionCount = 0;

	for (int i = 0; i < 8; i++) {
		int neighborIndex = neighbors[i];
		if (neighborIndex != -1) {
			int start = gridCellStartIndices[neighborIndex];
			int end = gridCellEndIndices[neighborIndex];
			if (start != -1 && end != -1) {
				for (int j = start; j <= end; j++) {
					if (j != index) {
						glm::vec3 otherPos = pos[j];

						float distance = glm::length(pos[index] - otherPos);

						// Rule 1 - Cohesion: boids fly towards their local perceived center of mass, which excludes themselves
						if (distance < rule1Distance) {
							cohesion += otherPos;
							++cohesionCount;
						}

						// Rule 2 - Separation: boids try to stay a distance d away from each other
						if (distance < rule2Distance) {
							separation -= otherPos - pos[index];
						}

						// Rule 3 - Alignment: boids try to match the speed of surrounding boids
						if (distance < rule3Distance) {
							alignment += vel1[j];
							++alignmentCount;
						}
					}
				}
			}
		}
	}

	// Average out the cohesion velocity and scale it
	if (cohesionCount > 0) {
		cohesion /= cohesionCount;
		cohesion = (cohesion - pos[index]) * rule1Scale;
		deltaVel += cohesion;
	}

	// Scale the separation velocity
	separation *= rule2Scale;
	deltaVel += separation;

	// Average out the cohesion velocity and scale it
	if (alignmentCount > 0) {
		alignment *= rule3Scale / alignmentCount;
		deltaVel += alignment;
	}

	glm::vec3 newVel = vel1[index] + deltaVel;
	float newSpeed = glm::length(newVel);
	newVel = newSpeed > maxSpeed ? (newVel / newSpeed) * maxSpeed : newVel;
	vel2[index] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);	

	// Computer new velocity (vel2)
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, dev_vel1, dev_vel2);

	// Update position of boids based on new velocity (vel2)
	kernUpdatePos << <fullBlocksPerGrid, blockSize>> >(numObjects, dt, dev_pos, dev_vel2);

	// Ping-pong the velocity buffers (from vel2 to vel1)
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;

	//cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationScatteredGrid(float dt) {
	// Compute grid indices & array indices
	dim3 fullBlocksPerGridObject((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGridObject, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
																   dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// Unstable sort keys (grid indices) & values (array indices)
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// Reset start & end buffers to indicate that cells that we won't traverse do not contain boids
	dim3 fullBlocksPerGridCellCount((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer << <fullBlocksPerGridCellCount, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGridCellCount, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);

	// Naively traverse the sorted key-value pairs and identify the start and end of cells
	kernIdentifyCellStartEnd << <fullBlocksPerGridObject, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	// Computer new velocity (vel2)
	kernUpdateVelNeighborSearchScattered << <fullBlocksPerGridObject, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
																					 dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, 
																					 dev_pos, dev_vel1, dev_vel2);

	// Update position of boids based on new velocity (vel2)
	kernUpdatePos << <fullBlocksPerGridObject, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);

	// Ping-pong the velocity buffers (from vel2 to vel1)
	cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationCoherentGrid(float dt) {
	// Compute grid indices & array indices
	dim3 fullBlocksPerGridObject((numObjects + blockSize - 1) / blockSize);
	kernComputeIndices << <fullBlocksPerGridObject, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
																   dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

	// Unstable sort keys (grid indices) & values (array indices)
	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

	// Reset start & end buffers to indicate that cells that we won't traverse do not contain boids
	dim3 fullBlocksPerGridCellCount((gridCellCount + blockSize - 1) / blockSize);
	kernResetIntBuffer << <fullBlocksPerGridCellCount, blockSize >> >(gridCellCount, dev_gridCellStartIndices, -1);
	kernResetIntBuffer << <fullBlocksPerGridCellCount, blockSize >> >(gridCellCount, dev_gridCellEndIndices, -1);

	// Naively traverse the sorted key-value pairs and identify the start and end of cells
	kernIdentifyCellStartEnd << <fullBlocksPerGridObject, blockSize >> >(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

	// Reshuffle data arrays
	kernShuffleDataArrays << <fullBlocksPerGridObject, blockSize >> > (dev_particleArrayIndices, dev_pos, dev_posCpy, dev_vel1, dev_vel1Cpy, dev_vel2, dev_vel2Cpy);
	cudaMemcpy(dev_pos, dev_posCpy, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_vel1, dev_vel1Cpy, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_vel2, dev_vel2Cpy, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);

	// Update velocity
	kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGridObject, blockSize >> >(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
																				    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos, dev_vel1, dev_vel2);

	// Update position of boids based on new velocity (vel2)
	kernUpdatePos << <fullBlocksPerGridObject, blockSize >> >(numObjects, dt, dev_pos, dev_vel2);

	// Ping-pong the velocity buffers (from vel2 to vel1)
	cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::endSimulation() {
	cudaFree(dev_vel1);
	cudaFree(dev_vel2);
	cudaFree(dev_pos);
	
	cudaFree(dev_particleArrayIndices);
	cudaFree(dev_particleGridIndices);
	cudaFree(dev_gridCellStartIndices);
	cudaFree(dev_gridCellEndIndices);

	cudaFree(dev_vel1Cpy);
	cudaFree(dev_vel2Cpy);
	cudaFree(dev_posCpy);
}

void Boids::unitTest() {
	// test unstable sort
	int *dev_intKeys;
	int *dev_intValues;
	int N = 10;

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}
