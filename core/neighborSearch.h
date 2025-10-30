#pragma once
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "deviceStructs.h"

void sortKeyValuePairs(int* keys, int* values, int num);

void inclusiveScan(int* prefixSum, int* count, int num);

__global__ void calculatePointHash(pointCloud pC, spatialGrid sG);

__global__ void setInitialIndices(int* initialIndexes, int numObjects);

__global__ void setHashAux(int* hashAux, int* hash, int numObjects);

__global__ void findStartAndEnd(int* start, int* end, int* hash, int* hashAux, int numObjects);

void buildHashSpans(int* start, int* end, int* sortedIndexes, int* hash, int* hashAux, int numObjects, int maxThreadsPerBlock);

void updateGridCellStartEnd(fluid& f, solid& s, spatialGrid& sG, int maxThreadsPerBlock);

__global__ void setFluidInteractionsKernel(interactionBase fluid2Fluid, interactionBase fluid2Solid, fluid f, solid s, spatialGrid sG, int flag);

__global__ void setSolid2SolidInteractionsKernel(interactionSolid2Solid solid2Solid, solid s, spatialGrid sG, int flag);

void neighborSearch(DeviceData& d, int iStep, int fluidNeighborSearchGap, int maxThreadsPerBlock);

inline void computeGPUParameter(int& gridSize, int& blockSize,
    int nElements,
    int maxThreadsPerBlock)
{
    if (nElements == 0)
    {
        gridSize = int(1);
        blockSize = int(1);
        return;
    }
    blockSize = maxThreadsPerBlock < nElements ? maxThreadsPerBlock : nElements;
    gridSize = (nElements + blockSize - 1) / blockSize;
}

__device__ __forceinline__ int3 calculateGridPosition(double3 position, double3 minBoundary, double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
        int((position.y - minBoundary.y) / cellSize.y),
        int((position.z - minBoundary.z) / cellSize.z));
}

__device__ __forceinline__ int calculateHash(int3 gridPosition, int3 gridSize)
{
    return gridPosition.z * gridSize.y * gridSize.x + gridPosition.y * gridSize.x + gridPosition.x;
}