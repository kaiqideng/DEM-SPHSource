#include "neighborSearch.h"

void sortKeyValuePairs(int* keys, int* values, int num)
{
    if (num < 1) return;
    thrust::sort_by_key(thrust::device_ptr<int>(keys),
        thrust::device_ptr<int>(keys + num),
        thrust::device_ptr<int>(values));
}

void inclusiveScan(int* prefixSum, int* count, int num)
{
    if (num < 1) return;
    thrust::inclusive_scan(thrust::device_ptr<int>(count),
        thrust::device_ptr<int>(count + num),
        thrust::device_ptr<int>(prefixSum));
}

__global__ void calculatePointHash(pointCloud pC,
    spatialGrid sG)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pC.num) return;
    double3 pos = pC.position[idx];
    if (sG.minBound.x <= pos.x && pos.x < sG.maxBound.x &&
        sG.minBound.y <= pos.y && pos.y < sG.maxBound.y &&
        sG.minBound.z <= pos.z && pos.z < sG.maxBound.z)
    {
        int3 gridPosition = calculateGridPosition(pos, sG.minBound, sG.cellSize);
        pC.hash.value[idx] = calculateHash(gridPosition, sG.gridSize);
    }
    else
    {
        pC.hash.value[idx] = sG.num - 1;
    }
}

__global__ void setInitialIndices(int* initialIndexes,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    initialIndexes[indices] = indices;
}

__global__ void setHashAux(int* hashAux,
    int* hash,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0) hashAux[0] = hash[numObjects - 1];
    if (indices > 0)  hashAux[indices] = hash[indices - 1];
}

__global__ void findStartAndEnd(int* start, int* end,
    int* hash,
    int* hashAux,
    int numObjects)
{
    int indices = blockIdx.x * blockDim.x + threadIdx.x;
    if (indices >= numObjects) return;
    if (indices == 0 || hash[indices] != hashAux[indices])
    {
        start[hash[indices]] = indices;
        end[hashAux[indices]] = indices;
    }
    if (indices == numObjects - 1) end[hash[indices]] = numObjects;
}

void buildHashSpans(int* start, int* end, int* sortedIndexes, int* hash, int* hashAux,
    int numObjects,
    int maxThreadsPerBlock)
{
    if (numObjects < 1) return;

    int grid = 1, block = 1;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);

    setInitialIndices << <grid, block >> > (sortedIndexes, numObjects);

    sortKeyValuePairs(hash, sortedIndexes, numObjects);

    setHashAux << <grid, block >> > (hashAux, hash, numObjects);

    findStartAndEnd << <grid, block >> > (start, end, hash, hashAux, numObjects);
}

void updateGridCellStartEnd(fluid& f, solid& s, spatialGrid& sG, int maxThreadsPerBlock)
{
    sG.resetCellStartEnd();
    f.points.hash.reset(f.points.num);
    s.points.hash.reset(s.points.num);

    int grid = 1, block = 1;
    int numObjects = 0;

    numObjects = f.points.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    calculatePointHash << <grid, block >> > (f.points, sG);

    numObjects = s.points.num;
    computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    calculatePointHash << <grid, block >> > (s.points, sG);

    buildHashSpans(sG.cellStartFluid, sG.cellEndFluid, f.points.hash.index, f.points.hash.value, f.points.hash.aux,
        f.points.num,
        maxThreadsPerBlock);

    buildHashSpans(sG.cellStartSolid, sG.cellEndSolid, s.points.hash.index, s.points.hash.value, s.points.hash.aux,
        s.points.num,
        maxThreadsPerBlock);
}

__global__ void setFluidInteractionsKernel(interactionBase fluid2Fluid, interactionBase fluid2Solid,
    fluid f,
	solid s,
    spatialGrid sG,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= f.points.num)  return;
    f.fluidNeighbor.count[idxA] = 0;
	f.solidNeighbor.count[idxA] = 0;
    if (flag == 0)
    {
        f.fluidNeighbor.prefixSum[idxA] = 0;
		f.solidNeighbor.prefixSum[idxA] = 0;
    }
    int fluidCount = 0, solidCount = 0;
    int fluidBase = (idxA == 0 ? 0 : f.fluidNeighbor.prefixSum[idxA - 1]), solidBase = (idxA == 0 ? 0 : f.solidNeighbor.prefixSum[idxA - 1]);
    double3 posA = f.points.position[idxA];
    double radA = f.points.effectiveRadii[idxA];
    int3 gridPositionA = calculateGridPosition(posA, sG.minBound, sG.cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, sG.gridSize);
                if (hashB < 0 || hashB >= sG.num)
                {
                    continue;
                }

                //fluid - fluid:
                int startIndex = sG.cellStartFluid[hashB];
                int endIndex = sG.cellEndFluid[hashB];
                if (startIndex == 0xFFFFFFFF)
                {
                    goto fluidSolid;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = f.points.hash.index[i];
                    if (idxA >= idxB) continue;
                    double3 posB = f.points.position[idxB];
                    double radB = f.points.effectiveRadii[idxB];
					double rad = fmax(radA, radB);
                    double3 rAB = posA - posB;
                    double dis = length(rAB);
                    double overlap = 2 * rad - dis;
                    if (overlap >= 0.)
                    {
                        if (flag == 0)
                        {
                            fluidCount++;
                        }
                        else
                        {
                            int offset = atomicAdd(&f.fluidNeighbor.count[idxA], 1);
                            int posWrite = fluidBase + offset;
                            fluid2Fluid.objectPointed[posWrite] = idxA;
                            fluid2Fluid.objectPointing[posWrite] = idxB;
                            fluid2Fluid.force[posWrite] = make_double3(0, 0, 0);
                        }
                    }
                }

                fluidSolid:
                startIndex = sG.cellStartSolid[hashB];
                endIndex = sG.cellEndSolid[hashB];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = s.points.hash.index[i];
                    double3 posB = s.points.position[idxB];
                    double3 rAB = posA - posB;
                    double dis = length(rAB);
                    double overlap = 2 * radA - dis;
                    if (overlap >= 0.)
                    {
                        if (flag == 0)
                        {
                            solidCount++;
                        }
                        else
                        {
                            int offset = atomicAdd(&f.solidNeighbor.count[idxA], 1);
                            int posWrite = solidBase + offset;
                            fluid2Solid.objectPointed[posWrite] = idxA;
                            fluid2Solid.objectPointing[posWrite] = idxB;
                            fluid2Solid.force[posWrite] = make_double3(0, 0, 0);
                        }
                    }
                }
            }
        }
    }
    if (flag == 0)
    {
        f.fluidNeighbor.count[idxA] = fluidCount;
		f.solidNeighbor.count[idxA] = solidCount;
    }
}

__global__ void setSolid2SolidInteractionsKernel(interactionSolid2Solid solid2Solid,
    solid s,
    spatialGrid sG,
    int flag)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= s.points.num)  return;
    if (s.isDummyParticle[idxA] == 1) return;
    s.solidNeighbor.count[idxA] = 0;
    if (flag == 0)
    {
        s.solidNeighbor.prefixSum[idxA] = 0;
    }
    int count = 0;
    int base = (idxA == 0 ? 0 : s.solidNeighbor.prefixSum[idxA - 1]);
    double3 posA = s.points.position[idxA];
    double radA = s.points.effectiveRadii[idxA];
    int3 gridPositionA = calculateGridPosition(posA, sG.minBound, sG.cellSize);
    for (int zz = -1; zz <= 1; zz++)
    {
        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = calculateHash(gridPositionB, sG.gridSize);
                if (hashB < 0 || hashB >= sG.num)
                {
                    continue;
                }
                int startIndex = sG.cellStartSolid[hashB];
                int endIndex = sG.cellEndSolid[hashB];
                if (startIndex == 0xFFFFFFFF)
                {
                    continue;
                }
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = s.points.hash.index[i];
                    if (s.isDummyParticle[idxB] == 1) continue;
					if (idxA >= idxB) continue;
                    if (s.clumpID[idxA] >= 0 && s.clumpID[idxA] == s.clumpID[idxB]) continue;
					if (s.inverseMass[idxA] < 1.e-10 && s.inverseMass[idxB] < 1.e-10) continue;
                    double3 posB = s.points.position[idxB];
                    double radB = s.points.effectiveRadii[idxB];
                    double3 rAB = posA - posB;
                    double dis = length(rAB);
                    double overlap = radB + radA - dis;
                    if (overlap >= 0.)
                    {
                        if (flag == 0)
                        {
                            count++;
                        }
                        else
                        {
                            int offset = atomicAdd(&s.solidNeighbor.count[idxA], 1);
                            int posWrite = base + offset;
                            solid2Solid.objectPointed[posWrite] = idxA;
                            solid2Solid.objectPointing[posWrite] = idxB;
                            solid2Solid.force[posWrite] = make_double3(0, 0, 0);
                            solid2Solid.torque[posWrite] = make_double3(0, 0, 0);
							solid2Solid.slidingSpring[posWrite] = make_double3(0, 0, 0);
							solid2Solid.rollingSpring[posWrite] = make_double3(0, 0, 0);
							solid2Solid.torsionSpring[posWrite] = make_double3(0, 0, 0);
                            if (s.solidRange.start[idxB] != 0xFFFFFFFF)
                            {
                                for (int j = s.solidRange.start[idxB]; j < s.solidRange.end[idxB]; j++)
                                {
                                    int j1 = solid2Solid.hash.index[j];
                                    int idxA1 = solid2Solid.history.objectPointed[j1];
                                    if (idxA == idxA1)
                                    {
                                        solid2Solid.slidingSpring[posWrite] = solid2Solid.history.slidingSpring[j1];
                                        solid2Solid.rollingSpring[posWrite] = solid2Solid.history.rollingSpring[j1];
                                        solid2Solid.torsionSpring[posWrite] = solid2Solid.history.torsionSpring[j1];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (flag == 0)
    {
        s.solidNeighbor.count[idxA] = count;
    }
}

void neighborSearch(DeviceData& d, int iStep, int fluidNeighborSearchGap, int maxThreadsPerBlock)
{
	updateGridCellStartEnd(d.fluids, d.solids, d.spatialGrids, maxThreadsPerBlock);

    int grid = 1, block = 1;
    int numObjects = 0;
    
    if (iStep % fluidNeighborSearchGap == 0)
    {
        numObjects = d.fluids.points.num;
        if (numObjects > 0)
        {
            computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
            for (int flag = 0; flag < 2; flag++)
            {
                setFluidInteractionsKernel << <grid, block >> > (d.fluid2Fluid, d.fluid2Solid,
                    d.fluids,
                    d.solids,
                    d.spatialGrids,
                    flag);
                if (flag == 0)
                {
                    int nFF = 0, nFS = 0;
                    inclusiveScan(d.fluids.fluidNeighbor.prefixSum, d.fluids.fluidNeighbor.count, numObjects);
                    inclusiveScan(d.fluids.solidNeighbor.prefixSum, d.fluids.solidNeighbor.count, numObjects);
                    cudaMemcpy(&nFF, d.fluids.fluidNeighbor.prefixSum + numObjects - 1, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&nFS, d.fluids.solidNeighbor.prefixSum + numObjects - 1, sizeof(int), cudaMemcpyDeviceToHost);
                    d.fluid2Fluid.setNum(nFF);
                    d.fluid2Solid.setNum(nFS);
                }
            }
            d.fluids.fluidRange.reset(numObjects);
            d.fluid2Fluid.hash.reset(d.fluid2Fluid.capacity);
            d.fluid2Fluid.setHash();
            buildHashSpans(d.fluids.fluidRange.start, d.fluids.fluidRange.end, d.fluid2Fluid.hash.index, d.fluid2Fluid.hash.value, d.fluid2Fluid.hash.aux, d.fluid2Fluid.num, maxThreadsPerBlock);

            numObjects = d.solids.points.num;
            d.solids.fluidSolidRange.reset(numObjects);
            d.fluid2Solid.hash.reset(d.fluid2Solid.capacity);
            d.fluid2Solid.setHash();
            buildHashSpans(d.solids.fluidSolidRange.start, d.solids.fluidSolidRange.end, d.fluid2Solid.hash.index, d.fluid2Solid.hash.value, d.fluid2Solid.hash.aux, d.fluid2Solid.num, maxThreadsPerBlock);
        }
    }
    
    numObjects = d.solids.points.num;
    if (numObjects > 0)
    {
        d.solid2Solid.save();//!!!!!!
        computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
        for (int flag = 0; flag < 2; flag++)
        {
            setSolid2SolidInteractionsKernel << <grid, block >> > (d.solid2Solid,
                d.solids,
                d.spatialGrids,
                flag);
            if (flag == 0)
            {
                int nSS = 0;
                inclusiveScan(d.solids.solidNeighbor.prefixSum, d.solids.solidNeighbor.count, numObjects);
                cudaMemcpy(&nSS, d.solids.solidNeighbor.prefixSum + numObjects - 1, sizeof(int), cudaMemcpyDeviceToHost);
                d.solid2Solid.setNum(nSS);
            }
        }
        d.solids.solidRange.reset(numObjects);
        d.solid2Solid.hash.reset(d.solid2Solid.capacity);
        d.solid2Solid.setHash();
        buildHashSpans(d.solids.solidRange.start, d.solids.solidRange.end, d.solid2Solid.hash.index, d.solid2Solid.hash.value, d.solid2Solid.hash.aux, d.solid2Solid.num, maxThreadsPerBlock);
    }
}