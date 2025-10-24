#pragma once
#include "neighborSearch.h"
#include "SPHKernelDEMContactModel.h"

__global__ void calDummyParticlePressureSmoothedVelocity(solid s, fluid f, interactionBase fluid2Solid, double3 g);

__global__ void densityReinitialization(fluid f, solid s, interactionBase fluid2Fluid, interactionBase fluid2Solid);

__global__ void solveMassConservationEquationDensityIntegrate(fluid f, solid s, interactionBase fluid2Fluid, interactionBase fluid2Solid, double3 g, double dt);

__global__ void solveMomentumConservationEquation(fluid f, solid s, interactionBase fluid2Fluid, interactionBase fluid2Solid, double3 g);

__global__ void calFluid2SolidForce(interactionBase fluid2Solid, fluid f, solid s, double3 g);

__global__ void fluidVelocityIntegrate(fluid f, double dt);

__global__ void fluidPositionIntegrate(fluid f, double dt);

void fluidIntegrate(DeviceData& d, double timeStep, int iStep, int integrateGap, int maxThreadsPerBlock);

__global__ void calSolidContactForceTorque(interactionSolid2Solid solid2Solid, solid s, solidContactModel contactModels, double timeStep);

__global__ void calSolidBondedForceTorque(interactionBonded solidBond2Solid, interactionSolid2Solid solid2Solid, solid s, solidContactModel contactModels, double dt);

__global__ void calSolidForceTorque(solid s, interactionSolid2Solid solid2Solid, interactionBase fluid2Solid, double3 g);

__global__ void calClumpForceTorque(clump clumps, solid s, double3 g);

__global__ void solidVelocityAngularVelocityIntegrate(solid s, double dt);

__global__ void solidPositionIntegrate(solid s, double dt);

__global__ void clumpVelocityAngularVelocityIntegrate(clump clumps, solid s, double dt);

__global__ void clumpPositionIntegrate(clump clumps, double dt);

void calSolidContactAfterFluidIntegrate(DeviceData& d, double timeStep, int maxThreadsPerBlock);

void solidIntegrateBeforeContact(DeviceData& d, double timeStep, int maxThreadsPerBlock);

void solidIntegrateAfterContact(DeviceData& d, double timeStep, int maxThreadsPerBlock);