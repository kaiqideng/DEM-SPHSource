#include "integrate.h"

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
//__device__ __forceinline__
//double atomicAddDouble(double* addr, double val)
//{
//	return atomicAdd(addr, val);
//}
//#else                                                   
//__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
//{
//	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
//	unsigned long long old = *addr_ull, assumed;
//
//	do {
//		assumed = old;
//		double  old_d = __longlong_as_double(assumed);
//		double  new_d = old_d + val;
//		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
//	} while (assumed != old);
//
//	return __longlong_as_double(old);
//}
//#endif
//
//__device__ __forceinline__ void atomicAddDouble3(double3* arr, int idx, const double3& v)   // arr[idx] += v;
//{
//	atomicAddDouble(&(arr[idx].x), v.x);
//	atomicAddDouble(&(arr[idx].y), v.y);
//	atomicAddDouble(&(arr[idx].z), v.z);
//}

__global__ void densityReinitialization(fluid f,
	solid s,
	interactionBase fluid2Fluid,
	interactionBase fluid2Solid)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= f.points.num) return;

	double3 r_i = f.points.position[idx_i];
	double h_i = f.points.effectiveRadii[idx_i];
	double sumW = wendlandKernel3D(0, h_i);

	for (int k = idx_i > 0 ? f.fluidNeighbor.prefixSum[idx_i - 1] : 0; k < f.fluidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_j = fluid2Fluid.objectPointing[k];
		double3 r_j = f.points.position[idx_j];
		double h_j = f.points.effectiveRadii[idx_j];
		double h = fmax(h_i, h_j);
		sumW += wendlandKernel3D(length(r_i - r_j), h);
	}

	if (f.fluidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = f.fluidRange.start[idx_i]; k < f.fluidRange.end[idx_i]; k++)
		{
			int k1 = fluid2Fluid.hash.index[k];
			int idx_j = fluid2Fluid.objectPointed[k1];
			double3 r_j = f.points.position[idx_j];
			double h_j = f.points.effectiveRadii[idx_j];
			double h = fmax(h_i, h_j);
			sumW += wendlandKernel3D(length(r_i - r_j), h);
		}
	}

	for (int k = idx_i > 0 ? f.solidNeighbor.prefixSum[idx_i - 1] : 0; k < f.solidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_a = fluid2Solid.objectPointing[k];
		double3 r_a = s.points.position[idx_a];
		double h = h_i;
		sumW += wendlandKernel3D(length(r_i - r_a), h);
	}

	if (f.sumW0[idx_i] < 1.e-10)
	{
		f.sumW0[idx_i] = sumW;
	}
	else
	{
		double rho0 = f.rho0[idx_i];
		double rho_star = rho0 + f.dRho[idx_i];
		double sumW0 = f.sumW0[idx_i];
		double rho1 = rho0 * sumW / sumW0 + fmax(0., rho_star - rho0 * sumW / sumW0) * rho0 / rho_star;
		f.dRho[idx_i] = rho1 - rho0;
	}
}

__global__ void solveMassConservationEquationDensityIntegrate(fluid f,
	solid s,
	interactionBase fluid2Fluid,
	interactionBase fluid2Solid,
	double3 g,
	double dt)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= f.points.num) return;

	double3 r_i = f.points.position[idx_i];
	double3 v_i = f.dyn.velocities[idx_i];
	double h_i = f.points.effectiveRadii[idx_i];
	double rho0_i = f.rho0[idx_i];
	double dRho_i = f.dRho[idx_i];
	double rho_i = rho0_i + dRho_i;
	double c_i = f.c[idx_i];
	double P_i = dRho_i * c_i * c_i;
	double dRho_dt = 0.;

	for (int k = idx_i > 0 ? f.fluidNeighbor.prefixSum[idx_i - 1] : 0; k < f.fluidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_j = fluid2Fluid.objectPointing[k];
		double3 r_j = f.points.position[idx_j];
		double3 v_j = f.dyn.velocities[idx_j];
		double h_j = f.points.effectiveRadii[idx_j];
		double dRho_j = f.dRho[idx_j];
		double rho_j = f.rho0[idx_j] + dRho_j;
		double c_j = f.c[idx_j];
		double P_j = dRho_j * c_j * c_j;
		double m_j = pow(h_j / 1.3, 3) * f.rho0[idx_j];
		double V_j = m_j / rho_j;

		double h = fmax(h_i, h_j);
		double c = fmax(c_i, c_j);
		double3 r_ij = r_i - r_j;
		double3 e_ij = normalize(r_ij);
		double3 deltaW_ij = gradWendlandKernel3D(r_ij, h);
		double U_L = dot(v_i, -e_ij);
		double U_R = dot(v_j, -e_ij);
		double P_L = P_i;
		double P_R = P_j;
		double rho_L = rho_i;
		double rho_R = rho_j;
		double U_star = (rho_L * U_L + rho_R * U_R + (P_L - P_R) / c) / (rho_L + rho_R);
		double3 v_star = (U_star - (rho_L * U_L + rho_R * U_R) / (rho_L + rho_R)) * (-e_ij) + (rho_i * v_i + rho_j * v_j) / (rho_i + rho_j);
		dRho_dt += 2 * rho_i * V_j * dot(v_i - v_star, deltaW_ij);
	}

	if (f.fluidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = f.fluidRange.start[idx_i]; k < f.fluidRange.end[idx_i]; k++)
		{
			int k1 = fluid2Fluid.hash.index[k];
			int idx_j = fluid2Fluid.objectPointed[k1];
			double3 r_j = f.points.position[idx_j];
			double3 v_j = f.dyn.velocities[idx_j];
			double h_j = f.points.effectiveRadii[idx_j];
			double dRho_j = f.dRho[idx_j];
			double rho_j = f.rho0[idx_j] + dRho_j;
			double c_j = f.c[idx_j];
			double P_j = dRho_j * c_j * c_j;
			double m_j = pow(h_j / 1.3, 3) * f.rho0[idx_j];
			double V_j = m_j / rho_j;

			double h = fmax(h_i, h_j);
			double c = fmax(c_i, c_j);
			double3 r_ij = r_i - r_j;
			double3 e_ij = normalize(r_ij);
			double3 deltaW_ij = gradWendlandKernel3D(r_ij, h);
			double U_L = dot(v_i, -e_ij);
			double U_R = dot(v_j, -e_ij);
			double P_L = P_i;
			double P_R = P_j;
			double rho_L = rho_i;
			double rho_R = rho_j;
			double U_star = (rho_L * U_L + rho_R * U_R + (P_L - P_R) / c) / (rho_L + rho_R);
			double3 v_star = (U_star - (rho_L * U_L + rho_R * U_R) / (rho_L + rho_R)) * (-e_ij) + (rho_i * v_i + rho_j * v_j) / (rho_i + rho_j);
			dRho_dt += 2 * rho_i * V_j * dot(v_i - v_star, deltaW_ij);
		}
	}

	for (int k = idx_i > 0 ? f.solidNeighbor.prefixSum[idx_i - 1] : 0; k < f.solidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_a = fluid2Solid.objectPointing[k];
		double3 r_a = s.points.position[idx_a];
		double3 v_a = s.dyn.velocities[idx_a];
		double V_a = 0;
		if (s.inverseMass[idx_a] > 0. && s.radius[idx_a] > 0.) V_a = 4. / 3. * pow(s.radius[idx_a], 3) * pi();
		else V_a = pow(h_i / 1.3, 3);

		double h = h_i;
		double c = c_i;
		double3 r_ia = r_i - r_a;
		double3 e_ia = normalize(r_ia);
		double3 deltaW_ia = gradWendlandKernel3D(r_ia, h);
		double3 dv_dt_a = s.dyn.accelerations[idx_a];
		double U_L = dot(v_i, -e_ia);
		double U_R = dot(2 * v_i - v_a, -e_ia);
		double P_L = P_i;
		double P_a = P_i + rho_i * fmax(0., dot(g - dv_dt_a, e_ia)) * dot(r_ia, e_ia);
		//double P_a = s.pressure[idx_a];
		double P_R = P_a;
		double rho_L = rho_i;
		double rho_a = P_a / (c * c) + rho0_i;
		double rho_R = rho_a;
		double U_star = (rho_L * U_L + rho_R * U_R + (P_L - P_R) / c) / (rho_L + rho_R);
		double3 v_star = (U_star - (rho_L * U_L + rho_R * U_R) / (rho_L + rho_R)) * (-e_ia) + (rho_i * v_i + rho_a * v_a) / (rho_i + rho_a);
		dRho_dt += 2 * rho_i * V_a * dot(v_i - v_star, deltaW_ia);
	}

	f.dRho[idx_i] += dt * dRho_dt;
}

__global__ void solveMomentumConservationEquation(fluid f,
	solid s,
	interactionBase fluid2Fluid,
	interactionBase fluid2Solid,
	double3 g)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= f.points.num) return;

	double3 r_i = f.points.position[idx_i];
	double3 v_i = f.dyn.velocities[idx_i];
	double h_i = f.points.effectiveRadii[idx_i];
	double rho0_i = f.rho0[idx_i];
	double dRho_i = f.dRho[idx_i];
	double rho_i = rho0_i + dRho_i;
	double c_i = f.c[idx_i];
	double P_i = dRho_i * c_i * c_i;
	double gamma_i = f.v[idx_i] * rho_i;
	double m_i = pow(h_i / 1.3, 3) * rho0_i;
	double V_i = m_i / rho_i;
	double3 dv_dt = make_double3(0, 0, 0);
	double3 f_s = make_double3(0, 0, 0);

	for (int k = idx_i > 0 ? f.fluidNeighbor.prefixSum[idx_i - 1] : 0; k < f.fluidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_j = fluid2Fluid.objectPointing[k];
		double3 r_j = f.points.position[idx_j];
		double3 v_j = f.dyn.velocities[idx_j];
		double h_j = f.points.effectiveRadii[idx_j];
		double dRho_j = f.dRho[idx_j];
		double rho_j = f.rho0[idx_j] + dRho_j;
		double c_j = f.c[idx_j];
		double P_j = dRho_j * c_j * c_j;
		double gamma_j = f.v[idx_j] * rho_j;
		double m_j = pow(h_j / 1.3, 3) * f.rho0[idx_j];
		double V_j = m_j / rho_j;

		double h = fmax(h_i, h_j);
		double c = fmax(c_i, c_j);
		double3 r_ij = r_i - r_j;
		double3 e_ij = normalize(r_ij);
		double3 deltaW_ij = gradWendlandKernel3D(r_ij, h);
		double U_L = dot(v_i, -e_ij);
		double U_R = dot(v_j, -e_ij);
		double P_L = P_i;
		double P_R = P_j;
		double rho_L = rho_i;
		double rho_R = rho_j;
		double beta = fmin(3. * fmax(U_L - U_R, 0.), c);
		double P_star = (rho_L * P_R + rho_R * P_L + rho_L * rho_R * beta * (U_L - U_R)) / (rho_L + rho_R);
		dv_dt += -2 / m_i * V_i * V_j * P_star * deltaW_ij + 2 / m_i * V_i * V_j * 2 * gamma_i * gamma_j / (gamma_i + gamma_j) * (v_i - v_j) / length(r_ij) * dot(e_ij, deltaW_ij);
	}

	if (f.fluidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = f.fluidRange.start[idx_i]; k < f.fluidRange.end[idx_i]; k++)
		{
			int k1 = fluid2Fluid.hash.index[k];
			int idx_j = fluid2Fluid.objectPointed[k1];
			double3 r_j = f.points.position[idx_j];
			double3 v_j = f.dyn.velocities[idx_j];
			double h_j = f.points.effectiveRadii[idx_j];
			double dRho_j = f.dRho[idx_j];
			double rho_j = f.rho0[idx_j] + dRho_j;
			double c_j = f.c[idx_j];
			double P_j = dRho_j * c_j * c_j;
			double gamma_j = f.v[idx_j] * rho_j;
			double m_j = pow(h_j / 1.3, 3) * f.rho0[idx_j];
			double V_j = m_j / rho_j;

			double h = fmax(h_i, h_j);
			double c = fmax(c_i, c_j);
			double3 r_ij = r_i - r_j;
			double3 e_ij = normalize(r_ij);
			double3 deltaW_ij = gradWendlandKernel3D(r_ij, h);
			double U_L = dot(v_i, -e_ij);
			double U_R = dot(v_j, -e_ij);
			double P_L = P_i;
			double P_R = P_j;
			double rho_L = rho_i;
			double rho_R = rho_j;
			double beta = fmin(3. * fmax(U_L - U_R, 0.), c);
			double P_star = (rho_L * P_R + rho_R * P_L + rho_L * rho_R * beta * (U_L - U_R)) / (rho_L + rho_R);
			dv_dt += -2 / m_i * V_i * V_j * P_star * deltaW_ij + 2 / m_i * V_i * V_j * 2 * gamma_i * gamma_j / (gamma_i + gamma_j) * (v_i - v_j) / length(r_ij) * dot(e_ij, deltaW_ij);
		}
	}

	for (int k = idx_i > 0 ? f.solidNeighbor.prefixSum[idx_i - 1] : 0; k < f.solidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_a = fluid2Solid.objectPointing[k];
		double3 r_a = s.points.position[idx_a];
		double3 v_a = s.dyn.velocities[idx_a];
		double V_a = 0;
		if (s.inverseMass[idx_a] > 0. && s.radius[idx_a] > 0.) V_a = 4. / 3. * pow(s.radius[idx_a], 3) * pi();
		else V_a = pow(h_i / 1.3, 3);

		double h = h_i;
		double c = c_i;
		double3 r_ia = r_i - r_a;
		double3 e_ia = normalize(r_ia);
		double3 deltaW_ia = gradWendlandKernel3D(r_ia, h);
		double3 dv_dt_a = s.dyn.accelerations[idx_a];
		double P_L = P_i;
		double P_a = P_i + rho_i * fmax(0., dot(g - dv_dt_a, e_ia)) * dot(r_ia, e_ia);
		//double P_a = s.pressure[idx_a];
		double P_R = P_a;
		double rho_L = rho_i;
		double rho_a = P_a / (c * c) + rho0_i;
		double rho_R = rho_a;
		double P_star = (rho_L * P_R + rho_R * P_L) / (rho_L + rho_R);
		double3 f_s0 = f_s;
		f_s += -2 / m_i * V_i * V_a * P_star * deltaW_ia + 2 / m_i * V_i * V_a * gamma_i * (v_i - v_a) / length(r_ia) * dot(e_ia, deltaW_ia);
		fluid2Solid.force[k] = (f_s - f_s0) * m_i;
	}

	dv_dt += f_s + g;
	f.dyn.accelerations[idx_i] = dv_dt;
}

__global__ void calDummyParticlePressureSmoothedVelocity(solid s,
	fluid f,
	interactionBase fluid2Solid,
	double3 g)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= s.points.num) return;

	s.pressure[idx_i] = 0.;
	s.smoothedVelocity[idx_i] = make_double3(0, 0, 0);
	double item0 = 0.; double3 item1 = make_double3(0, 0, 0), item11 = make_double3(0, 0, 0); double item2 = 0.;
	double3 r_w = s.points.position[idx_i];
	double3 a_w = s.dyn.accelerations[idx_i];
	if (s.fluidSolidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = s.fluidSolidRange.start[idx_i]; k < s.fluidSolidRange.end[idx_i]; k++)
		{
			int k1 = fluid2Solid.hash.index[k];
			int idx_j = fluid2Solid.objectPointed[k1];
			double3 v_f = f.dyn.velocities[idx_j];
			double3 r_f = f.points.position[idx_j];
			double h = f.points.effectiveRadii[idx_j];
			double rho_f = f.rho0[idx_j] + f.dRho[idx_j];
			double p_f = pow(f.c[idx_j], 2) * f.dRho[idx_j];
			double W = wendlandKernel3D(length(r_w - r_f), h);
			item0 += p_f * W;
			item1 += (r_w - r_f) * rho_f * W;
			item11 += v_f * W;
			item2 += W;
		}
	}

	if (item2 > 1.e-10)
	{
		s.pressure[idx_i] = (item0 + dot(g - a_w, item1)) / item2;
		s.smoothedVelocity[idx_i] = 2. * s.dyn.velocities[idx_i] - item11 / item2;
	}
}

__global__ void fluidVelocityIntegrate(fluid f,
	double dt)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= f.points.num) return;

	f.dyn.velocities[idx_i] += dt * f.dyn.accelerations[idx_i];
}

__global__ void fluidPositionIntegrate(fluid f,
	double dt)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= f.points.num) return;

	f.points.position[idx_i] += dt * f.dyn.velocities[idx_i];
}

void fluidIntegrate(DeviceData& d, double timeStep, int iStep, int integrateGap, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = 0;

	//numObjects = d.solids.points.num;
    //computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
    //calDummyParticlePressureSmoothedVelocity << <grid, block >> > (d.solids, d.fluids, d.fluid2Solid, d.gravity);

	numObjects = d.fluids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	double timeStep_f = 0;
	int run2ndIntegration = 0;
	if (iStep % integrateGap == 0)
	{
		timeStep_f = timeStep* double(integrateGap);
		run2ndIntegration = 1;

		if (iStep % (5 * integrateGap) == 0) densityReinitialization << <grid, block >> > (d.fluids, d.solids, d.fluid2Fluid, d.fluid2Solid);

		solveMassConservationEquationDensityIntegrate << <grid, block >> > (d.fluids, d.solids, d.fluid2Fluid, d.fluid2Solid, d.gravity, 0.5 * timeStep_f);

		fluidPositionIntegrate << <grid, block >> > (d.fluids, 0.5 * timeStep_f);
	}

	solveMomentumConservationEquation << <grid, block >> > (d.fluids, d.solids, d.fluid2Fluid, d.fluid2Solid, d.gravity);

	if (run2ndIntegration == 1)
	{
		fluidVelocityIntegrate << <grid, block >> > (d.fluids, timeStep_f);

		fluidPositionIntegrate << <grid, block >> > (d.fluids, 0.5 * timeStep_f);

		solveMassConservationEquationDensityIntegrate << <grid, block >> > (d.fluids, d.solids, d.fluid2Fluid, d.fluid2Solid, d.gravity, 0.5 * timeStep_f);
	}
}

__global__ void calSolidContactForceTorque(interactionSolid2Solid solid2Solid,
	solid s,
	solidContactModel contactModels,
	double dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= solid2Solid.num) return;

	solid2Solid.force[idx] = make_double3(0, 0, 0);
	solid2Solid.torque[idx] = make_double3(0, 0, 0);

	int idx_i = solid2Solid.objectPointed[idx];
	int idx_j = solid2Solid.objectPointing[idx];
	double rad_i = s.radius[idx_i];
	double rad_j = s.radius[idx_j];
	double3 r_i = s.points.position[idx_i];
	double3 r_j = s.points.position[idx_j];
	double3 n_ij = normalize(r_i - r_j);
	double delta = rad_i + rad_j - length(r_i - r_j);
	double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
	double rad_ij = rad_i * rad_j / (rad_i + rad_j);
	double m_ij = 1. / (s.inverseMass[idx_i] + s.inverseMass[idx_j]);

	double3 v_i = s.dyn.velocities[idx_i];
	double3 v_j = s.dyn.velocities[idx_j];
	double3 w_i = s.angularVelocities[idx_i];
	double3 w_j = s.angularVelocities[idx_j];
	double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	double3 epsilon_s = solid2Solid.slidingSpring[idx];
	double3 epsilon_r = solid2Solid.rollingSpring[idx];
	double3 epsilon_t = solid2Solid.torsionSpring[idx];

	int mat_i = s.materialID[idx_i];
	int mat_j = s.materialID[idx_j];
	int c_ij = contactModels.getCombinedIndex(mat_i, mat_j);

	if (contactModels.linear.k_n[c_ij] > 1.e-10)
	{
		LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij,
			w_ij,
			n_ij,
			delta,
			m_ij,
			rad_ij,
			dt,
			contactModels.linear.k_n[c_ij],
			contactModels.linear.k_s[c_ij],
			contactModels.linear.k_r[c_ij],
			contactModels.linear.k_t[c_ij],
			contactModels.linear.d_n[c_ij],
			contactModels.linear.d_s[c_ij],
			contactModels.linear.d_r[c_ij],
			contactModels.linear.d_t[c_ij],
			contactModels.linear.mu_s[c_ij],
			contactModels.linear.mu_r[c_ij],
			contactModels.linear.mu_t[c_ij]);
	}
	else
	{
		double logR = log(contactModels.hertzian.res[c_ij]);
		double D = -logR / sqrt(logR * logR + pi() * pi());
		HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij,
			w_ij,
			n_ij,
			delta,
			m_ij,
			rad_ij,
			dt,
			contactModels.hertzian.E[c_ij],
			contactModels.hertzian.G[c_ij],
			D,
			contactModels.hertzian.k_r_k_s[c_ij],
			contactModels.hertzian.k_t_k_s[c_ij],
			contactModels.hertzian.mu_s[c_ij],
			contactModels.hertzian.mu_r[c_ij],
			contactModels.hertzian.mu_t[c_ij]);
	}

	solid2Solid.force[idx] = F_c;
	solid2Solid.torque[idx] = T_c;
	solid2Solid.slidingSpring[idx] = epsilon_s;
	solid2Solid.rollingSpring[idx] = epsilon_r;
	solid2Solid.torsionSpring[idx] = epsilon_t;
}

__global__ void calSolidBondedForceTorque(interactionBonded solidBond2Solid,
	interactionSolid2Solid solid2Solid,
	solid s,
	solidContactModel contactModels,
	double dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= solidBond2Solid.num) return;

	if (solidBond2Solid.isBonded[idx] == 0)
	{
		solidBond2Solid.normalForce[idx] = 0;
		solidBond2Solid.torsionTorque[idx] = 0;
		solidBond2Solid.shearForce[idx] = make_double3(0, 0, 0);
		solidBond2Solid.bendingTorque[idx] = make_double3(0, 0, 0);
		solidBond2Solid.contactNormal[idx] = make_double3(0, 0, 0);
		return;
	}

	int idx_i = solidBond2Solid.objectPointed[idx];
	int idx_j = solidBond2Solid.objectPointing[idx];

	bool find = false;
	int idx_solid2Solid = 0;
	int neighborStart_i = idx_i > 0 ? s.solidNeighbor.prefixSum[idx_i - 1] : 0;
	int neighborEnd_i = s.solidNeighbor.prefixSum[idx_i];
	for (int k = neighborStart_i; k < neighborEnd_i; k++)
	{
		if (solid2Solid.objectPointing[k] == idx_j)
		{
			find = true;
			idx_solid2Solid = k;
			break;
		}
	}
	if (!find)
	{
		solidBond2Solid.isBonded[idx] = 0;
		return;
	}

	double rad_i = s.radius[idx_i];
	double rad_j = s.radius[idx_j];
	double3 r_i = s.points.position[idx_i];
	double3 r_j = s.points.position[idx_j];
	double3 n_ij = normalize(r_i - r_j);
	double3 n_ij0 = solidBond2Solid.contactNormal[idx];
	double delta = rad_i + rad_j - length(r_i - r_j);
	double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
	double3 v_i = s.dyn.velocities[idx_i];
	double3 v_j = s.dyn.velocities[idx_j];
	double3 w_i = s.angularVelocities[idx_i];
	double3 w_j = s.angularVelocities[idx_j];
	double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

	int mat_i = s.materialID[idx_i];
	int mat_j = s.materialID[idx_j];
	int c_ij = contactModels.getCombinedIndex(mat_i, mat_j);
	double F_n = solidBond2Solid.normalForce[idx];
	double3 F_s = solidBond2Solid.shearForce[idx];
	double T_t = solidBond2Solid.torsionTorque[idx];
	double3 T_b = solidBond2Solid.bendingTorque[idx];
	solidBond2Solid.isBonded[idx] = ParallelBondedContact(F_n, T_t, F_s, T_b,
		n_ij0,
		n_ij,
		v_c_ij,
		w_i,
		w_j,
		rad_i,
		rad_j,
		dt,
		contactModels.bonded.gamma[c_ij],
		contactModels.bonded.E[c_ij],
		contactModels.bonded.k_n_k_s[c_ij],
		contactModels.bonded.sigma_s[c_ij],
		contactModels.bonded.C[c_ij],
		contactModels.bonded.mu[c_ij]);

	solidBond2Solid.normalForce[idx] = F_n;
	solidBond2Solid.shearForce[idx] = F_s;
	solidBond2Solid.torsionTorque[idx] = T_t;
	solidBond2Solid.bendingTorque[idx] = T_b;
	solidBond2Solid.contactNormal[idx] = n_ij;
	solidBond2Solid.contactPoint[idx] = r_c;

	solid2Solid.force[idx_solid2Solid] += F_n * n_ij + F_s;
	solid2Solid.torque[idx_solid2Solid] += T_t * n_ij + T_b;
}

__global__ void calSolidForceTorque(solid s,
	interactionSolid2Solid solid2Solid,
	interactionBase fluid2Solid,
	double3 g)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= s.points.num) return;

	double rad_i = s.radius[idx_i];
	double3 r_i = s.points.position[idx_i];
	double3 F_i = make_double3(0, 0, 0);
	double3 T_i = make_double3(0, 0, 0);
	for (int k = idx_i > 0 ? s.solidNeighbor.prefixSum[idx_i - 1] : 0; k < s.solidNeighbor.prefixSum[idx_i]; k++)
	{
		int idx_j = solid2Solid.objectPointing[k];
		double rad_j = s.radius[idx_j];
		double3 r_j = s.points.position[idx_j];
		double3 n_ij = normalize(r_i - r_j);
		double delta = rad_i + rad_j - length(r_i - r_j);
		double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
		F_i += solid2Solid.force[k];
		T_i += solid2Solid.torque[k] + cross(r_c - r_i, solid2Solid.force[k]);
	}

	if (s.solidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = s.solidRange.start[idx_i]; k < s.solidRange.end[idx_i]; k++)
		{
			int k1 = solid2Solid.hash.index[k];
			int idx_j = solid2Solid.objectPointed[k1];
			double rad_j = s.radius[idx_j];
			double3 r_j = s.points.position[idx_j];
			double3 n_ij = normalize(r_i - r_j);
			double delta = rad_i + rad_j - length(r_i - r_j);
			double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
			F_i -= solid2Solid.force[k1];
			T_i -= solid2Solid.torque[k1];
			T_i -= cross(r_c - r_i, solid2Solid.force[k1]);
		}
	}

	if (s.fluidSolidRange.start[idx_i] != 0xFFFFFFFF)
	{
		for (int k = s.fluidSolidRange.start[idx_i]; k < s.fluidSolidRange.end[idx_i]; k++)
		{
			int k1 = fluid2Solid.hash.index[k];
			F_i -= fluid2Solid.force[k1];
		}
	}

	s.dyn.accelerations[idx_i] = F_i * s.inverseMass[idx_i] + g * (s.inverseMass[idx_i] > 0. && s.clumpID[idx_i] < 0);
	s.torques[idx_i] = T_i;
}

__global__ void calClumpForceTorque(clump clumps,
	solid s,
	double3 g)
{
	int idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= clumps.num) return;
	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	for (int i = clumps.pebbleStartIndex[idx_c]; i < clumps.pebbleEndIndex[idx_c]; i++)
	{
		double3 r_i = s.points.position[i];
		double3 F_i = make_double3(0, 0, 0);
		if (s.inverseMass[i] > 0.) F_i = s.dyn.accelerations[i] / s.inverseMass[i];
		double3 r_c = clumps.centroidPosition[idx_c];
		F_c += F_i;
		T_c += s.torques[i] + cross(r_i - r_c, F_i);
		s.dyn.accelerations[i] = make_double3(0, 0, 0);
		s.torques[i] = make_double3(0, 0, 0);
	}
	clumps.dyn.accelerations[idx_c] = F_c * clumps.inverseMass[idx_c] + g * (clumps.inverseMass[idx_c] > 0.);
	clumps.torques[idx_c] = T_c;
}

__global__ void solidVelocityAngularVelocityIntegrate(solid s,
	double dt)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= s.points.num) return;

	if (s.clumpID[idx_i] >= 0) return;
	s.dyn.velocities[idx_i] += s.dyn.accelerations[idx_i] * dt;
	double rad_i = s.radius[idx_i];
	double invM_i = s.inverseMass[idx_i];
	if (invM_i < 1.e-10 || rad_i < 1.e-10) return;
	double3 T_i = s.torques[idx_i];
	double I_i = 0.4 * rad_i * rad_i / invM_i;
	s.angularVelocities[idx_i] += T_i / I_i * dt;
}

__global__ void solidPositionIntegrate(solid s,
	double dt)
{
	int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= s.points.num) return;

	s.points.position[idx_i] += dt * s.dyn.velocities[idx_i];
}

__global__ void clumpVelocityAngularVelocityIntegrate(clump clumps,
	solid s,
	double dt)
{
	int idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= clumps.num) return;

	double3 a_c = clumps.dyn.accelerations[idx_c];
	double3 w_c = clumps.angularVelocities[idx_c];
	clumps.dyn.velocities[idx_c] += a_c * dt;
	double invM_c = clumps.inverseMass[idx_c];
	if (invM_c > 0.) clumps.angularVelocities[idx_c] += (rotateInverseInertiaTensor(clumps.orientations[idx_c], clumps.inverseInertiaTensor[idx_c]) * clumps.torques[idx_c]) * dt;

	for (int i = clumps.pebbleStartIndex[idx_c]; i < clumps.pebbleEndIndex[idx_c]; i++)
	{
		double3 r_pc = s.points.position[i] - clumps.centroidPosition[idx_c];
		s.dyn.accelerations[i] = a_c + cross(a_c, r_pc) + cross(w_c, cross(w_c, r_pc));
		s.dyn.velocities[i] = clumps.dyn.velocities[idx_c] + cross(w_c, r_pc);
		s.angularVelocities[i] = w_c;
	}
}

__global__ void clumpPositionIntegrate(clump clumps,
	double dt)
{
	int idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= clumps.num) return;

	clumps.centroidPosition[idx_c] += dt * clumps.dyn.velocities[idx_c];
}

void calSolidContactAfterFluidIntegrate(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.solid2Solid.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calSolidContactForceTorque << <grid, block >> > (d.solid2Solid, d.solids, d.contactModels, timeStep);

	numObjects = d.solidBond2Solid.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calSolidBondedForceTorque << <grid, block >> > (d.solidBond2Solid, d.solid2Solid, d.solids, d.contactModels, timeStep);

	numObjects = d.solids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calSolidForceTorque << <grid, block >> > (d.solids, d.solid2Solid, d.fluid2Solid, d.gravity);

	numObjects = d.clumps.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	calClumpForceTorque << <grid, block >> > (d.clumps, d.solids, d.gravity);
}

void solidIntegrateBeforeContact(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.solids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	solidVelocityAngularVelocityIntegrate << <grid, block >> > (d.solids, 0.5 * timeStep);

	numObjects = d.clumps.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	clumpVelocityAngularVelocityIntegrate << <grid, block >> > (d.clumps, d.solids, 0.5 * timeStep);

	clumpPositionIntegrate << <grid, block >> > (d.clumps, 0.5 * timeStep);

	numObjects = d.solids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	solidPositionIntegrate << <grid, block >> > (d.solids, 0.5 * timeStep);
}

void solidIntegrateAfterContact(DeviceData& d, double timeStep, int maxThreadsPerBlock)
{
	int grid = 1, block = 1;
	int numObjects = d.solids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	solidPositionIntegrate << <grid, block >> > (d.solids, 0.5 * timeStep);

	numObjects = d.clumps.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	clumpPositionIntegrate << <grid, block >> > (d.clumps, 0.5 * timeStep);

	clumpVelocityAngularVelocityIntegrate << <grid, block >> > (d.clumps, d.solids, 0.5 * timeStep);

	numObjects = d.solids.points.num;
	computeGPUParameter(grid, block, numObjects, maxThreadsPerBlock);
	solidVelocityAngularVelocityIntegrate << <grid, block >> > (d.solids, 0.5 * timeStep);

}



