#include "solverBase.h"

class damBreak :public solverBase
{
public:
	damBreak() : solverBase() {}

	double spacing = 0.025;

	void conditionInitialize() override
	{
		setProblemName("damBreak");

		setDomain(make_double3(-3. * spacing, -3. * spacing, -3. * spacing), make_double3(8 + 6. * spacing, 0.7 + 6. * spacing, 0.7 + 6. * spacing));

		std::vector<double3> fp = getRegularPackedPoints(make_double3(0, 0, 0), make_double3(3.5, 0.7, 0.4), spacing);
		addFluid(fp, make_double3(0, 0, 0), 1.3 * spacing, 1000., 30., 1.e-6);

		std::vector<double3> sp0 = getRegularPackedPoints(make_double3(-3 * spacing, -3 * spacing, -3 * spacing), make_double3(8 + 6 * spacing, 0.7 + 6 * spacing, 0.8 + 6 * spacing), spacing);
		std::vector<double3> sp1;
		for (const auto& p : sp0)
		{
			if (p.x < 0 || p.y < 0 || p.z < 0 || p.x > 8. || p.y > 0.7) sp1.push_back(p);
		}
		addSolid(sp1, make_double3(0, 0, 0), 0.5 * spacing, 0., 0);

		setHertzianContactModel(0, 1, 3e9, 3e9 / (2 * (1 + 0.3)), 0.9, 0, 0, 0.35, 0, 0);
		setHertzianContactModel(1, 1, 3e9, 3e9 / (2 * (1 + 0.3)), 0.9, 0, 0, 0.45, 0, 0);
		double r_ball = 0.015;
		double k = 0.5 * 3e9 * pi() * r_ball;
		double mij = 4. / 3. * pi() * r_ball * r_ball * r_ball * 800.;
		double dt = 0.2 * sqrt(mij / k);
		double dt_f = 0.25 * 1.3 * spacing / (20 * sqrt(0.4 * 9.8));
		int gap = int(dt_f / dt);
		if (gap < 1) gap = 1;
		setSimulationParameterTimeStep(dt);
		setFluidIntegrateGap(gap);

		setSimulationParameterMaximumTime(2.);
		setSimulationParameterNumFrames(100);
	}

	void handleDataAfterCalculateContact() override
	{
		double r_ball = 0.015;
		double k = 0.5 * 3e9 * pi() * r_ball;
		double mij = 4. / 3. * pi() * r_ball * r_ball * r_ball * 800.;
		double dt = 0.2 * sqrt(mij / k);
		if (getTime() >= 0.5 && getTime() < 0.5 + dt)
		{
			double3 s_cub = make_double3(0.15, 0.15, 0.15);
			std::vector<double3> sp2 = getRegularPackedPoints(make_double3(5.3, 0.275, 0.), s_cub, 2 * r_ball);
			std::vector<double3> sp3 = getRegularPackedPoints(make_double3(5.3, 0.275, 0.15), s_cub, 2 * r_ball);
			std::vector<double3> sp4 = getRegularPackedPoints(make_double3(5.3, 0.275, 0.3), s_cub, 2 * r_ball);
			std::vector<double> rad(sp2.size(), r_ball);
			double mass = 800 * s_cub.x * s_cub.y * s_cub.z;
			symMatrix inertia = make_symMatrix(mass / 12. * (s_cub.y * s_cub.y + s_cub.z * s_cub.z), mass / 12. * (s_cub.x * s_cub.x + s_cub.z * s_cub.z), mass / 12. * (s_cub.x * s_cub.x + s_cub.y * s_cub.y), 0., 0., 0.);
			addClump(sp2, rad, make_double3(5.375, 0.35, 0.075), make_double3(0, 0, 0), mass, inertia, 1);
			addClump(sp3, rad, make_double3(5.375, 0.35, 0.225), make_double3(0, 0, 0), mass, inertia, 1);
			addClump(sp4, rad, make_double3(5.375, 0.35, 0.375), make_double3(0, 0, 0), mass, inertia, 1);
		}
	}

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