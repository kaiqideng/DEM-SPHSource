#pragma once
#include "integrate.h"
#include "fileEditingTool.h"
#include "generatePoints.h"

struct simulationParameter
{
    double maximumTime;
    double timeStep;
    int numSteps;
    int iStep;
    int numFrames;
    int iFrame;;
    int frameInterval;
};

struct GPUParameter
{
    int deviceIndex;
    int maxThreadsPerBlock;
};

class data
{
public:
    data()
    {
        simPara.maximumTime = 1.;
        simPara.timeStep = 1.;
        simPara.numSteps = 1;
        simPara.iStep = 0;
        simPara.numFrames = 1;
        simPara.iFrame = 0;
        simPara.frameInterval = 1;
        gpuPara.deviceIndex = 0;
        gpuPara.maxThreadsPerBlock = 256;
        domainOrigin = make_double3(0, 0, 0);
        domainSize = make_double3(1, 1, 1);
        gravity = make_double3(0, 0, -9.81);
        fluidIntegrateGap = 1;
    }

    ~data()
    {
        dev.release();
    }

protected:

    void setGPUParameterDeviceIndex(int index)
    {
        gpuPara.deviceIndex = index;
    }

    void setSimulationParameterMaximumTime(double t)
    {
        if (t <= 0.)
        {
			std::cout << "Error: maximum time must be positive." << std::endl;
			return;
        }
        if (simPara.iStep > 0)
        {
            std::cout << "Warning: cannot change the number of frames during simulation." << std::endl;
            return;
        }
        simPara.maximumTime = t;
    }

    void setSimulationParameterNumFrames(int n)
    {
		if (n < 1)
		{
			std::cout << "Error: number of frames must be at least 1." << std::endl;
			return;
		}
        if (simPara.iStep > 0)
        {
			std::cout << "Warning: cannot change the number of frames during simulation." << std::endl;
			return;
        }
        simPara.numFrames = n;
    }

    void setSimulationParameterTimeStep(double dt)
    {
        if (dt <= 0.)
        {
			std::cout << "Error: time step must be non-negative." << std::endl;
			return;
        }
        if (simPara.iStep > 0)
        {
			std::cout << "Warning: cannot change the time step during simulation." << std::endl;
            return;
        }
		simPara.timeStep = dt;
    }

    void setFluidIntegrateGap(int gap)
    {
        if (gap < 1)
        {
            std::cout << "Error: fluid integrate gap must be at least 1." << std::endl;
            return;
        }
		fluidIntegrateGap = gap;
    }

    void setDomain(double3 origin, double3 size)
    {
        domainOrigin = origin;
        domainSize = size;
		if (size.x <= 0. || size.y <= 0. || size.z <= 0.)
		{
			std::cout << "Error: domain size must be positive in all directions." << std::endl;
			return;
		}
        if (simPara.iStep > 0) setSpatialGrids();
    }

    void setGravity(double3 g)
    {
        gravity = g;
        if (simPara.iStep > 0) dev.gravity = gravity;
    }

    void setHertzianContactModel(int mat_i, int mat_j, double E, double G, double res, double k_r_k_s, double k_t_k_s, double mu_s, double mu_r, double mu_t)
    {
		int c_ij = hos.contactModels.getCombinedIndex(mat_i, mat_j);
		if (c_ij < 0)
		{
			std::cout << "Error: material index exceeds the number of materials." << std::endl;
			return;
		}
		hos.contactModels.hertzian.E[c_ij] = E;
		hos.contactModels.hertzian.G[c_ij] = G;
		hos.contactModels.hertzian.res[c_ij] = res;
		hos.contactModels.hertzian.k_r_k_s[c_ij] = k_r_k_s;
		hos.contactModels.hertzian.k_t_k_s[c_ij] = k_t_k_s;
		hos.contactModels.hertzian.mu_s[c_ij] = mu_s;
		hos.contactModels.hertzian.mu_r[c_ij] = mu_r;
		hos.contactModels.hertzian.mu_t[c_ij] = mu_t;
		if (simPara.iStep > 0) dev.contactModels.copy(hos.contactModels);
    }

	void setLinearContactModel(int mat_i, int mat_j, double k_n, double k_s, double k_r, double k_t, double d_n, double d_s, double d_r, double d_t, double mu_s, double mu_r, double mu_t)
	{
		int c_ij = hos.contactModels.getCombinedIndex(mat_i, mat_j);
        if (c_ij < 0)
        {
            std::cout << "Error: material index exceeds the number of materials." << std::endl;
            return;
        }
		hos.contactModels.linear.k_n[c_ij] = k_n;
		hos.contactModels.linear.k_s[c_ij] = k_s;
		hos.contactModels.linear.k_r[c_ij] = k_r;
		hos.contactModels.linear.k_t[c_ij] = k_t;
		hos.contactModels.linear.d_n[c_ij] = d_n;
		hos.contactModels.linear.d_s[c_ij] = d_s;
		hos.contactModels.linear.d_r[c_ij] = d_r;
		hos.contactModels.linear.d_t[c_ij] = d_t;
		hos.contactModels.linear.mu_s[c_ij] = mu_s;
		hos.contactModels.linear.mu_r[c_ij] = mu_r;
		hos.contactModels.linear.mu_t[c_ij] = mu_t;
        if (simPara.iStep > 0) dev.contactModels.copy(hos.contactModels);
	}

	void setBondedContactModel(int mat_i, int mat_j, double E, double k_n_k_s, double gamma, double sigma_s, double C, double mu)
	{
		int c_ij = hos.contactModels.getCombinedIndex(mat_i, mat_j);
        if (c_ij < 0)
        {
            std::cout << "Error: material index exceeds the number of materials." << std::endl;
            return;
        }
		hos.contactModels.bonded.E[c_ij] = E;
		hos.contactModels.bonded.k_n_k_s[c_ij] = k_n_k_s;
		hos.contactModels.bonded.gamma[c_ij] = gamma;
		hos.contactModels.bonded.sigma_s[c_ij] = sigma_s;
		hos.contactModels.bonded.C[c_ij] = C;
		hos.contactModels.bonded.mu[c_ij] = mu;
		if (simPara.iStep > 0) dev.contactModels.copy(hos.contactModels);
	}

    void addFluid(std::vector<double3> p, double3 velocity, double smoothLength, double density, double soundSpeed, double kinematicViscosity);

    void addSolid(std::vector<double3> p, double3 velocity, double radius, double density, int materialID);

    void addSPHSolidBoundary(std::vector<double3> p, double radius);

	void addCluster(std::vector<double3> p, std::vector<double> radius, std::vector<double> density, double3 velocity, int materialID);

    void addClump(std::vector<double3> p, std::vector<double> radius, double3 centroidPosition, double3 velocity, double mass, symMatrix inertiaTensor, int materialID);

    void addExternalForce(int index, double3 force);

    void addGlobalDamping(int index, double C_d);

	double getTime() const
	{
		return simPara.iStep * simPara.timeStep;
	}

	const HostSolid& getHostSolidData()
	{ 
        if (simPara.iStep < 1) return hos.solids;
		dev.solids.uploadState(hos.solids);
		return hos.solids;
	}

    const HostFluid& getFluidData()
    {
        if (simPara.iStep < 1) return hos.fluids;
        dev.fluids.uploadState(hos.fluids);
        return hos.fluids;
    }

private:
    HostData hos;
    DeviceData dev;
    simulationParameter simPara;
    GPUParameter gpuPara;
    double3 domainOrigin;
    double3 domainSize;
    double3 gravity;
	int fluidIntegrateGap;

    void addFluidData(const HostFluid f);

    void addSolidData(const HostSolid s);

    void addBondData();

    void setSolidNormal();

    void setSpatialGrids();

    void buildDeviceData();

	friend class solverBase;

    double3 gradWendlandKernel3DForSolidNormal(const double3& rij, double h)
    {
        double r = length(rij);
        if (r < 1.e-10 || r >= 2.0 * h) return make_double3(0, 0, 0);
        double q = r / h;
        double sigma = 21.0 / (16.0 * pi() * h * h * h);
        double term = 1.0 - 0.5 * q;
        double dW_dq = (-2.0 * pow(term, 3) * (1.0 + 2.0 * q) + 2.0 * pow(term, 4));
        double dWdr = sigma * dW_dq / h;
        double factor = dWdr / r;
        return factor * rij;
    }
};