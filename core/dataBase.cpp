#include "dataBase.h"

void dataBase::addFluid(std::vector<double3> p, double3 velocity, double smoothLength, double density, double soundSpeed, double kinematicViscosity)
{
    HostFluid f(int(p.size()));
    for (size_t i = 0; i < p.size();i++)
    {
        f.points.position[i] = p[i];
        f.points.effectiveRadii[i] = smoothLength;
        f.dyn.velocities[i] = velocity;
        f.rho0[i] = density;
        f.c[i] = soundSpeed;
        f.nu[i] = kinematicViscosity;
    }

    addFluidData(f);
}

void dataBase::addSolid(std::vector<double3> p, double3 velocity, double radius, double density, int materialID)
{
    if (materialID >= hos.contactModels.nMaterial)
    {
        std::cout << "Error: material index exceeds the number of materials when adding a cluster." << std::endl;
        return;
    }

    HostSolid s(int(p.size()));
    for (size_t i = 0; i < p.size();i++)
    {
        s.points.position[i] = p[i];
        s.dyn.velocities[i] = velocity;
        s.points.effectiveRadii[i] = 1.1 * radius;
        s.radius[i] = radius;
        s.materialID[i] = materialID;
        if (density > 0) s.inverseMass[i] = 1. / (4. / 3. * pow(radius, 3) * pi() * density);
    }

    addSolidData(s);
}

void dataBase::addFixedSolid(std::vector<double3> p, double radius, int materialID)
{

    HostSolid s(int(p.size()));
    for (size_t i = 0; i < p.size();i++)
    {
        s.points.position[i] = p[i];
        s.points.effectiveRadii[i] = 2.6 * radius;
        s.radius[i] = radius;
    }

    addSolidData(s);
}

void dataBase::addCluster(std::vector<double3> p, std::vector<double> radius, std::vector<double> density, double3 velocity, int materialID)
{
    if (p.size() != radius.size() || p.size() != density.size())
    {
        std::cout << "Error: the size of position, velocity, and radius vectors must be the same when adding a cluster." << std::endl;
        return;
    }
    if (materialID >= hos.contactModels.nMaterial)
    {
        std::cout << "Error: material index exceeds the number of materials when adding a cluster." << std::endl;
        return;
    }

    int clusterID = 0;
    if (hos.solids.points.num > 0) clusterID = *std::max_element(hos.solids.clusterID.begin(), hos.solids.clusterID.end()) + 1;
    HostSolid s(int(p.size()));
    for (size_t i = 0; i < p.size();i++)
    {
        s.points.position[i] = p[i];
        double physicalRadius = radius[i];
        s.points.effectiveRadii[i] = 1.1 * physicalRadius;
        s.dyn.velocities[i] = velocity;
        s.radius[i] = physicalRadius;
        s.materialID[i] = materialID;
        s.clusterID[i] = clusterID;
        if (density[i] > 0) s.inverseMass[i] = 1. / (4. / 3. * pow(physicalRadius, 3) * pi() * density[i]);
    }

    addSolidData(s);
    if (simPara.iStep > 0)
    {
        neighborSearch(dev, 0, 1, gpuPara.maxThreadsPerBlock);
        addBondData();
    }
}

void dataBase::addClump(std::vector<double3> p, std::vector<double> radius, double3 centroidPosition, double3 velocity, double3 angularVelocity, double mass, symMatrix inertiaTensor, int materialID)
{
    if (p.size() != radius.size())
    {
        std::cout << "Error: the size of position and radius vectors must be the same when adding a clump." << std::endl;
        return;
    }
    if (materialID >= hos.contactModels.nMaterial)
    {
        std::cout << "Error: material index exceeds the number of materials when adding a clump." << std::endl;
        return;
    }

    int clumpID = hos.clumps.num;
    HostClump c(1);
    c.centroidPosition[0] = centroidPosition;
    c.dyn.velocities[0] = velocity;
    c.angularVelocities[0] = angularVelocity;
    if (mass > 0.) c.inverseMass[0] = 1. / mass;
    c.inverseInertiaTensor[0] = inverse(inertiaTensor);
    c.pebbleStartIndex[0] = hos.solids.points.num;

    double volume = 0;
    for (size_t i = 0; i < p.size();i++)
    {
        volume += 4. / 3. * pi() * pow(radius[i], 3);
    }
    double density = 0;
    if (volume > 0.) density = mass / volume;

    HostSolid s(int(p.size()));
    for (size_t i = 0; i < p.size();i++)
    {
        s.points.position[i] = p[i];
        double physicalRadius = radius[i];
        s.points.effectiveRadii[i] = 1.1 * physicalRadius;
        s.dyn.velocities[i] = velocity + cross(angularVelocity, p[i] - centroidPosition);
        s.angularVelocities[i] = angularVelocity;
        s.radius[i] = physicalRadius;
        s.materialID[i] = materialID;
        s.clumpID[i] = clumpID;
        if (density > 0. && physicalRadius > 0) s.inverseMass[i] = 1. / (4. / 3. * pow(physicalRadius, 3) * pi() * density);
    }

    addSolidData(s);
    if (simPara.iStep == 0)
    {
        c.pebbleEndIndex[0] = hos.solids.points.num;
        hos.clumps.insertData(c);
    }
    else
    {
        dev.clumps.uploadState(hos.clumps);
        c.pebbleEndIndex[0] = hos.solids.points.num;
        hos.clumps.insertData(c);
        dev.clumps.copy(hos.clumps);
    }
}

void dataBase::addExternalForce(int index, double3 force)
{
    if (index < 0 || index >= hos.solids.points.num)
    {
        std::cout << "Error: the index of the solid particle is out of range when adding external force." << std::endl;
        return;
    }
    if (simPara.iStep == 0)
    {
        std::cout << "Error: external forces can only be added after the simulation starts." << std::endl;
        return;
    }

    cuda_copy(&hos.solids.dyn.accelerations[index], dev.solids.dyn.accelerations + index, 1, CopyDir::D2H);
    hos.solids.dyn.accelerations[index] += force * hos.solids.inverseMass[index];
    cuda_copy(dev.solids.dyn.accelerations + index, &hos.solids.dyn.accelerations[index], 1, CopyDir::H2D);
}

void dataBase::addGlobalDamping(int index, double C_d)
{
    if (index < 0 || index >= hos.solids.points.num)
    {
        std::cout << "Error: the index of the solid particle is out of range when adding global damping." << std::endl;
        return;
    }
    if (simPara.iStep == 0)
    {
        std::cout << "Error: global damping can only be added after the simulation starts." << std::endl;
        return;
    }

    double invM = hos.solids.inverseMass[index];
    if (invM < 1.e-10) return;
    cuda_copy(&hos.solids.dyn.accelerations[index], dev.solids.dyn.accelerations + index, 1, CopyDir::D2H);
    cuda_copy(&hos.solids.dyn.velocities[index], dev.solids.dyn.velocities + index, 1, CopyDir::D2H);
    cuda_copy(&hos.solids.torques[index], dev.solids.torques + index, 1, CopyDir::D2H);
    cuda_copy(&hos.solids.angularVelocities[index], dev.solids.angularVelocities + index, 1, CopyDir::D2H);
    double3 F = hos.solids.dyn.accelerations[index] / invM;
    double3 T = hos.solids.torques[index];
    if (length(hos.solids.dyn.velocities[index]) > 1.e-10) F += -C_d * length(F) * normalize(hos.solids.dyn.velocities[index]);
    if (length(hos.solids.angularVelocities[index]) > 1.e-10) T += -C_d * length(T) * normalize(hos.solids.angularVelocities[index]);
    hos.solids.dyn.accelerations[index] = F * invM;
    hos.solids.torques[index] = T;
    cuda_copy(dev.solids.dyn.accelerations + index, &hos.solids.dyn.accelerations[index], 1, CopyDir::H2D);
    cuda_copy(dev.solids.torques + index, &hos.solids.torques[index], 1, CopyDir::H2D);
}

void dataBase::addFluidData(const HostFluid f)
{
    if (simPara.iStep > 0)
    {
        dev.fluids.uploadState(hos.fluids);
        hos.fluids.insertData(f);
        dev.fluids.copy(hos.fluids);
        dev.fluid2Fluid.upload(hos.fluid2Fluid);
        hos.fluid2Fluid.insertData(HostInteractionBase(40 * f.points.num));
        dev.fluid2Fluid.copy(hos.fluid2Fluid);
        dev.fluid2Solid.upload(hos.fluid2Solid);
        hos.fluid2Solid.insertData(HostInteractionBase(40 * f.points.num));
        dev.fluid2Solid.copy(hos.fluid2Solid);
        setSpatialGrids();
    }
    else
    {
        hos.fluids.insertData(f);
        hos.fluid2Fluid.insertData(HostInteractionBase(40 * f.points.num));
        hos.fluid2Solid.insertData(HostInteractionBase(40 * f.points.num));
    }
}

void dataBase::addSolidData(const HostSolid s)
{
    if (simPara.iStep > 0)
    {
        dev.solids.uploadState(hos.solids);
        hos.solids.insertData(s);
        dev.solids.copy(hos.solids);
        dev.solid2Solid.upload(hos.solid2Solid);
        hos.solid2Solid.insertData(HostInteractionSolid2Solid(6 * s.points.num));
        dev.solid2Solid.copy(hos.solid2Solid);
        setSpatialGrids();
    }
    else
    {
        hos.solids.insertData(s);
        hos.solid2Solid.insertData(HostInteractionSolid2Solid(6 * s.points.num));
    }
}

void dataBase::addBondData()
{
    if (dev.solid2Solid.num > 0)
    {
        dev.solid2Solid.upload(hos.solid2Solid);
        if (simPara.iStep == 0)
        {
            for (int k = 0; k < hos.solid2Solid.num; k++)
            {
                int idx_i = hos.solid2Solid.objectPointed[k];
                int idx_j = hos.solid2Solid.objectPointing[k];
                double3 pos_i = hos.solids.points.position[idx_i];
                double3 pos_j = hos.solids.points.position[idx_j];
                double rad_i = hos.solids.radius[idx_i];
                double rad_j = hos.solids.radius[idx_j];
                if (rad_i + rad_j + 1.e-10 <= length(pos_i - pos_j)) continue;
                int mat_i = hos.solids.materialID[idx_i];
                int mat_j = hos.solids.materialID[idx_j];
                int cluster_i = hos.solids.clusterID[idx_i];
                int cluster_j = hos.solids.clusterID[idx_j];
                int c_ij = hos.contactModels.getCombinedIndex(mat_i, mat_j);
                double E_b = hos.contactModels.bonded.E[c_ij];
                if (E_b > 1.e-10 && cluster_i >= 0 && cluster_i == cluster_j)
                {
                    double d = length(pos_i - pos_j);
                    double3 n_ij = (pos_i - pos_j) / d;
                    double delta = rad_i + rad_j - d;
                    double3 r_c = pos_j + (rad_j - 0.5 * delta) * n_ij;
                    HostInteractionBonded bond(1);
                    bond.contactNormal[0] = n_ij;
                    bond.contactPoint[0] = r_c;
                    bond.objectPointed[0] = idx_i;
                    bond.objectPointing[0] = idx_j;
                    bond.isBonded[0] = 1;
                    hos.solidBond2Solid.insertData(bond);
                }
            }
            dev.solidBond2Solid.copy(hos.solidBond2Solid);
        }
        else
        {
            dev.solidBond2Solid.upload(hos.solidBond2Solid);
            for (int k = 0; k < hos.solid2Solid.num; k++)
            {
                int idx_i = hos.solid2Solid.objectPointed[k];
                if (idx_i <= hos.solidBond2Solid.objectPointed.back()) continue;
                int idx_j = hos.solid2Solid.objectPointing[k];
                double3 pos_i = hos.solids.points.position[idx_i];
                double3 pos_j = hos.solids.points.position[idx_j];
                double rad_i = hos.solids.radius[idx_i];
                double rad_j = hos.solids.radius[idx_j];
                if (rad_i + rad_j + 1.e-10 <= length(pos_i - pos_j)) continue;
                int mat_i = hos.solids.materialID[idx_i];
                int mat_j = hos.solids.materialID[idx_j];
                int cluster_i = hos.solids.clusterID[idx_i];
                int cluster_j = hos.solids.clusterID[idx_j];
                int c_ij = hos.contactModels.getCombinedIndex(mat_i, mat_j);
                double E_b = hos.contactModels.bonded.E[c_ij];
                if (E_b > 1.e-10 && cluster_i >= 0 && cluster_i == cluster_j)
                {
                    double d = length(pos_i - pos_j);
                    double3 n_ij = (pos_i - pos_j) / d;
                    double delta = rad_i + rad_j - d;
                    double3 r_c = pos_j + (rad_j - 0.5 * delta) * n_ij;
                    HostInteractionBonded bond(1);
                    bond.contactNormal[0] = n_ij;
                    bond.contactPoint[0] = r_c;
                    bond.objectPointed[0] = idx_i;
                    bond.objectPointing[0] = idx_j;
                    bond.isBonded[0] = 1;
                    hos.solidBond2Solid.insertData(bond);
                }
            }
            dev.solidBond2Solid.copy(hos.solidBond2Solid);
        }
    }
}

void dataBase::setSolidNormal()
{
    double3 minBound = dev.spatialGrids.minBound;
    double3 cellSize = dev.spatialGrids.cellSize;
    int3 gridSize = dev.spatialGrids.gridSize;
    int nGrids = dev.spatialGrids.num;
    std::vector<int> cellStart(nGrids, -1), cellEnd(nGrids, -1);
    cuda_copy(cellStart.data(), dev.spatialGrids.cellStartSolid, nGrids, CopyDir::D2H);
    cuda_copy(cellEnd.data(), dev.spatialGrids.cellEndSolid, nGrids, CopyDir::D2H);
    std::vector<int> hashIndex(hos.solids.points.num, -1);
    cuda_copy(hashIndex.data(), dev.solids.points.hash.index, hos.solids.points.num, CopyDir::D2H);
    double h = *std::max_element(hos.fluids.points.effectiveRadii.begin(), hos.fluids.points.effectiveRadii.end());
    for (int idxA = 0; idxA < hos.solids.points.num; idxA++)
    {
        double3 Theta = make_double3(0, 0, 0);

        int clumpIDA = hos.solids.clumpID[idxA];
        double invMassA = hos.solids.inverseMass[idxA];
        double3 posA = hos.solids.points.position[idxA];
        int3 gridPositionA = make_int3(int((posA.x - minBound.x) / cellSize.x), int((posA.y - minBound.y) / cellSize.y),
            int((posA.z - minBound.z) / cellSize.z));
        for (int zz = -1; zz <= 1; zz++)
        {
            for (int yy = -1; yy <= 1; yy++)
            {
                for (int xx = -1; xx <= 1; xx++)
                {
                    int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                    int hashB = gridPositionB.z * gridSize.y * gridSize.x + gridPositionB.y * gridSize.x + gridPositionB.x;
                    if (hashB < 0 || hashB >= nGrids)
                    {
                        continue;
                    }
                    int startIndex = cellStart[hashB];
                    int endIndex = cellEnd[hashB];
                    if (startIndex == 0xFFFFFFFF)
                    {
                        continue;
                    }
                    for (int k = startIndex; k < endIndex; k++)
                    {
                        int idxB = hashIndex[k];
                        double3 posB = hos.solids.points.position[idxB];
                        double3 rAB = posA - posB;
                        if (length(rAB) > 2 * h) continue;
                        if (invMassA <= 1.e-10 && hos.solids.inverseMass[idxB] <= 1.e-10)
                        {
                            double V_B = pow(h / 1.3, 3);
                            double3 gradW = gradWendlandKernel3DForSolidNormal(rAB, h);
                            Theta += -V_B * gradW;
                        }
                        else if (clumpIDA >= 0 && clumpIDA == hos.solids.clumpID[idxB])
                        {
                            double V_B = pow(2. * hos.solids.radius[idxB], 3.);
                            double3 gradW = gradWendlandKernel3DForSolidNormal(rAB, h);
                            Theta += -V_B * gradW;
                        }
                    }
                }
            }
        }
        if (length(Theta) > 1.e-10) hos.solids.normal[idxA] = normalize(Theta);
        else hos.solids.normal[idxA] = make_double3(0., 0., 0.);
    }
    cuda_copy(dev.solids.normal, hos.solids.normal.data(), dev.solids.points.num, CopyDir::H2D);
}

void dataBase::setSpatialGrids()
{
    dev.spatialGrids.release();
    double maxFluidEffectiveRadii = 0.;
    if (hos.fluids.points.num > 0) maxFluidEffectiveRadii = *std::max_element(hos.fluids.points.effectiveRadii.begin(), hos.fluids.points.effectiveRadii.end());
    double maxSolidEffectiveRadii = 0;
    if (hos.solids.points.num > 0) maxSolidEffectiveRadii = *std::max_element(hos.solids.points.effectiveRadii.begin(), hos.solids.points.effectiveRadii.end());
    double cellSizeOneDim = 2 * std::max(maxFluidEffectiveRadii, maxSolidEffectiveRadii);
    dev.spatialGrids.minBound = domainOrigin;
    dev.spatialGrids.maxBound = domainOrigin + domainSize;
    dev.spatialGrids.gridSize.x = domainSize.x > cellSizeOneDim ? int(domainSize.x / cellSizeOneDim) : 1;
    dev.spatialGrids.gridSize.y = domainSize.y > cellSizeOneDim ? int(domainSize.y / cellSizeOneDim) : 1;
    dev.spatialGrids.gridSize.z = domainSize.z > cellSizeOneDim ? int(domainSize.z / cellSizeOneDim) : 1;
    dev.spatialGrids.cellSize.x = domainSize.x / double(dev.spatialGrids.gridSize.x);
    dev.spatialGrids.cellSize.y = domainSize.y / double(dev.spatialGrids.gridSize.y);
    dev.spatialGrids.cellSize.z = domainSize.z / double(dev.spatialGrids.gridSize.z);
    dev.spatialGrids.alloc(dev.spatialGrids.gridSize.x * dev.spatialGrids.gridSize.y * dev.spatialGrids.gridSize.z + 1);
}

void dataBase::buildDeviceData()
{
    dev.release();
    setSpatialGrids();
    dev.gravity = gravity;
    dev.fluids.copy(hos.fluids);
    dev.solids.copy(hos.solids);
    dev.clumps.copy(hos.clumps);
    dev.fluid2Fluid.copy(hos.fluid2Fluid);
    dev.fluid2Solid.copy(hos.fluid2Solid);
    dev.solid2Solid.copy(hos.solid2Solid);
    dev.contactModels.copy(hos.contactModels);
}


