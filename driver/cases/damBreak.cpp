#include "solverBase.h"

class damBreak :public solverBase
{
public:
	damBreak() : solverBase() {}

	double particle_spacing = 0.025;
	double particle_radius = 0.5 * particle_spacing;
	double E = 0.3e9;
	double3 cubeSize = make_double3(0.15, 0.15, 0.15);
	double cubeDensity = 800;

	void conditionInitialize() override
	{
		setProblemName("damBreak");

		setDomain(make_double3(-3. * particle_spacing, -3. * particle_spacing, -3. * particle_spacing), make_double3(8 + 6. * particle_spacing, 0.7 + 6. * particle_spacing, 0.8 + 6. * particle_spacing));

		std::vector<double3> fp = getRegularPackedPoints(make_double3(0, 0, 0), make_double3(3.5, 0.7, 0.4), particle_spacing);
		addFluid(fp, make_double3(0, 0, 0), 1.3 * particle_spacing, 1000., 30., 1.e-6);

		std::vector<double3> sp0 = getRegularPackedPoints(make_double3(-3 * particle_spacing, -3 * particle_spacing, -3 * particle_spacing), make_double3(8 + 6 * particle_spacing, 0.7 + 6 * particle_spacing, 0.8 + 6 * particle_spacing), particle_spacing);
		std::vector<double3> sp1;
		for (const auto& p : sp0)
		{
			if (p.x < 0 || p.y < 0 || p.z < 0 || p.x > 8. || p.y > 0.7) sp1.push_back(p);
		}
		addDummyParticleWall(sp1, 0.5 * particle_spacing);
		addInfiniteWall(make_double3(4, 0.35, 0), make_double3(0, 0, 1), 0);

		setHertzianContactModel(0, 1, E, 0.3 * E, 0.9, 0, 0, 0.35, 0, 0);
		setHertzianContactModel(1, 1, 0.5 * E, 0.15 * E, 0.9, 0, 0, 0.45, 0, 0);

		std::vector<double3> sp2 = getRegularPackedPoints(make_double3(5.2, 0.275, 0.), cubeSize, 2 * particle_radius);
		std::vector<double3> sp3 = getRegularPackedPoints(make_double3(5.2, 0.275, 0.15), cubeSize, 2 * particle_radius);
		std::vector<double3> sp4 = getRegularPackedPoints(make_double3(5.2, 0.275, 0.3), cubeSize, 2 * particle_radius);
		std::vector<double3> sp5 = getRegularPackedPoints(make_double3(3.5, 0., 0.), make_double3(3 * particle_spacing, 0.7, 0.8), 2 * particle_radius);
		std::vector<double> rad(sp2.size(), particle_radius);
		double mass = cubeDensity * cubeSize.x * cubeSize.y * cubeSize.z;
		symMatrix inertia = make_symMatrix(mass / 12. * (cubeSize.y * cubeSize.y + cubeSize.z * cubeSize.z), mass / 12. * (cubeSize.x * cubeSize.x + cubeSize.z * cubeSize.z), mass / 12. * (cubeSize.x * cubeSize.x + cubeSize.y * cubeSize.y), 0., 0., 0.);
		addClump(sp2, rad, make_double3(5.275, 0.35, 0.075), make_double3(0, 0, 0), make_double3(0, 0, 0), mass, inertia, 1);
		//addClump(sp3, rad, make_double3(5.275, 0.35, 0.225), make_double3(0, 0, 0), make_double3(0, 0, 0), mass, inertia, 1);
		//addClump(sp4, rad, make_double3(5.275, 0.35, 0.375), make_double3(0, 0, 0), make_double3(0, 0, 0), mass, inertia, 1);
		std::vector<double> rad1(sp5.size(), particle_radius);
		//addClump(sp5, rad1, make_double3(3.5 + 1.5 * particle_spacing, 0.35, 0.4), make_double3(0, 0, 0), make_double3(0, 0, 0), 0, make_symMatrix(0, 0, 0, 0, 0, 0), 0);
		double k = 0.5 * E * pi() * particle_radius;
		double mij = 8 * particle_radius * particle_radius * particle_radius * cubeDensity;
		double dt = 0.2 * sqrt(mij / k);
		double dt_f = 0.6 * 1.3 * particle_spacing / (22. * sqrt(0.4 * 9.8));
		int gap = int(dt_f / dt);
		if (gap < 1) gap = 1;
		setSimulationParameterTimeStep(dt);
		setFluidIntegrateGap(gap);

		setSimulationParameterMaximumTime(3.);
		setSimulationParameterNumFrames(150);
	}

	/*void handleDataAfterCalculateContact() override
	{
		double k = 0.5 * E * pi() * particle_radius;
		double mij = 8 * particle_radius * particle_radius * particle_radius * cubeDensity;
		double dt = 0.2 * sqrt(mij / k);
		if (getTime() >= 1. && getTime() < 1. + dt)
		{
			setClumpVelocity(1, make_double3(0, 0, 1));
		}
	}*/

	void outputData() override
	{
		outputFluidVTU();
		outputSolidVTU();
	}
};

int main()
{
	damBreak problem;
	problem.solve();
}