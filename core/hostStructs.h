#pragma once
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "myMat.h"

struct HostDynamicStateBase
{
	std::vector<double3>  velocities;
	std::vector<double3>  accelerations;

	HostDynamicStateBase() = default;

	explicit HostDynamicStateBase(int n)
	{
		velocities.resize(n, make_double3(0, 0, 0));
		accelerations.resize(n, make_double3(0, 0, 0));
	}

	void insertData(HostDynamicStateBase state)
	{
		velocities.insert(velocities.end(), state.velocities.begin(), state.velocities.end());
		accelerations.insert(accelerations.end(), state.accelerations.begin(), state.accelerations.end());
	}
};

struct HostPointCloud
{
	int num{ 0 };
	std::vector<double3> position;
	std::vector<double> effectiveRadii; // for collision detection

	HostPointCloud() = default;

	explicit HostPointCloud(int n)
		: num(n), position(n, make_double3(0, 0, 0)), effectiveRadii(n, 0)
	{
	}

	void insertData(HostPointCloud pc)
	{
		position.insert(position.end(), pc.position.begin(), pc.position.end());
		effectiveRadii.insert(effectiveRadii.end(), pc.effectiveRadii.begin(), pc.effectiveRadii.end());
		num += pc.num;
	}
};

struct HostFluid
{
	HostPointCloud points;
	HostDynamicStateBase dyn;
	std::vector<double> rho0; // initial density
	std::vector<double> dRho; // density change
	std::vector<double> c; // speed of sound
	std::vector<double> nu; // kinematic viscosity
	std::vector<double> chi; // background pressure term

	HostFluid() = default;

	explicit HostFluid(int n)
		: points(n), dyn(n), rho0(n, 0.0), dRho(n, 0.0), c(n, 0.0), nu(n, 0.0), chi(n, 0.0)
	{
	}

	void insertData(const HostFluid f)
	{
		points.insertData(f.points);
		dyn.insertData(f.dyn);
		rho0.insert(rho0.end(), f.rho0.begin(), f.rho0.end());
		dRho.insert(dRho.end(), f.dRho.begin(), f.dRho.end());
		c.insert(c.end(), f.c.begin(), f.c.end());
		nu.insert(nu.end(), f.nu.begin(), f.nu.end());
		chi.insert(chi.end(), f.chi.begin(), f.chi.end());
	}
};

struct HostSolid
{
	HostPointCloud points;
	HostDynamicStateBase dyn;
	std::vector<double3> normal;
	std::vector<double3> torques;
	std::vector<double3> angularVelocities;
	std::vector<double> radius;
	std::vector<double> inverseMass;
	std::vector<int> materialID;
	std::vector<int> clusterID;
	std::vector<int> clumpID;

	std::vector<double> pressure;

	HostSolid() = default;

	explicit HostSolid(int n)
		: points(n), dyn(n), normal(n, make_double3(0, 0, 0)), 
		torques(n, make_double3(0, 0, 0)), angularVelocities(n, make_double3(0, 0, 0)), radius(n, 0), inverseMass(n, 0), materialID(n, 0), clusterID(n, -1), clumpID(n, -1), pressure(n, 0)
	{
	}

	void insertData(const HostSolid s)
	{
		points.insertData(s.points);
		dyn.insertData(s.dyn);
		normal.insert(normal.end(), s.normal.begin(), s.normal.end());
		torques.insert(torques.end(), s.torques.begin(), s.torques.end());
		angularVelocities.insert(angularVelocities.end(), s.angularVelocities.begin(), s.angularVelocities.end());
		radius.insert(radius.end(), s.radius.begin(), s.radius.end());
		inverseMass.insert(inverseMass.end(), s.inverseMass.begin(), s.inverseMass.end());
		materialID.insert(materialID.end(), s.materialID.begin(), s.materialID.end());
		clusterID.insert(clusterID.end(), s.clusterID.begin(), s.clusterID.end());
		clumpID.insert(clumpID.end(), s.clumpID.begin(), s.clumpID.end());

		pressure.insert(pressure.end(), s.pressure.begin(), s.pressure.end());
	}
};

struct HostClump 
{
	int num{ 0 };

	HostDynamicStateBase dyn;
	std::vector<double3> centroidPosition;
	std::vector<double3> torques;
	std::vector<double3> angularVelocities;
	std::vector<quaternion> orientations;
	std::vector<symMatrix> inverseInertiaTensor;
	std::vector<double> inverseMass;
	std::vector<int> pebbleStartIndex;
	std::vector<int> pebbleEndIndex;

	HostClump() = default;
	explicit HostClump(int n)
		: num(n), dyn(n), centroidPosition(n, make_double3(0, 0, 0)), torques(n, make_double3(0, 0, 0)), angularVelocities(n, make_double3(0, 0, 0)), orientations(n, make_quaternion(1, 0, 0, 0)), inverseInertiaTensor(n, make_symMatrix(0, 0, 0, 0, 0, 0)), inverseMass(n, 0), pebbleStartIndex(n, -1), pebbleEndIndex(n, -1)
	{
	}

	void insertData(const HostClump c)
	{
		dyn.insertData(c.dyn);
		centroidPosition.insert(centroidPosition.end(), c.centroidPosition.begin(), c.centroidPosition.end());
		torques.insert(torques.end(), c.torques.begin(), c.torques.end());
		angularVelocities.insert(angularVelocities.end(), c.angularVelocities.begin(), c.angularVelocities.end());
		orientations.insert(orientations.end(), c.orientations.begin(), c.orientations.end());
		inverseInertiaTensor.insert(inverseInertiaTensor.end(), c.inverseInertiaTensor.begin(), c.inverseInertiaTensor.end());
		inverseMass.insert(inverseMass.end(), c.inverseMass.begin(), c.inverseMass.end());
		pebbleStartIndex.insert(pebbleStartIndex.end(), c.pebbleStartIndex.begin(), c.pebbleStartIndex.end());
		pebbleEndIndex.insert(pebbleEndIndex.end(), c.pebbleEndIndex.begin(), c.pebbleEndIndex.end());
		num += c.num;
	}
};

struct HostInteractionBase
{
	int capacity{ 0 }, num{ 0 };
	std::vector<double3> force;
	std::vector<int> objectPointed;
	std::vector<int> objectPointing;

	HostInteractionBase() = default;

	explicit HostInteractionBase(int n)
		: capacity(n), num(0), force(n, make_double3(0, 0, 0)), objectPointed(n, -1), objectPointing(n, -1)
	{
	}

	void insertData(const HostInteractionBase i)
	{
		force.insert(force.end(), i.force.begin(), i.force.end());
		objectPointed.insert(objectPointed.end(), i.objectPointed.begin(), i.objectPointed.end());
		objectPointing.insert(objectPointing.end(), i.objectPointing.begin(), i.objectPointing.end());
		capacity += i.capacity;
	}
};

struct HostInteractionSolid2Solid
{
	int capacity{ 0 }, num{ 0 };
	std::vector<double3> force;
	std::vector<double3> torque;
	std::vector<double3> slidingSpring;
	std::vector<double3> rollingSpring;
	std::vector<double3> torsionSpring;
	std::vector<int> objectPointed;
	std::vector<int> objectPointing;

	HostInteractionSolid2Solid() = default;

	explicit HostInteractionSolid2Solid(int n)
		: capacity(n), num(0), force(n, make_double3(0, 0, 0)),torque(n, make_double3(0, 0, 0)), slidingSpring(n, make_double3(0, 0, 0)), rollingSpring(n, make_double3(0, 0, 0)), torsionSpring(n, make_double3(0, 0, 0)), objectPointed(n, -1), objectPointing(n, -1)
	{
	}

	void insertData(const HostInteractionSolid2Solid i)
	{
		force.insert(force.end(), i.force.begin(), i.force.end());
		torque.insert(torque.end(), i.torque.begin(), i.torque.end());
		slidingSpring.insert(slidingSpring.end(), i.slidingSpring.begin(), i.slidingSpring.end());
		rollingSpring.insert(rollingSpring.end(), i.rollingSpring.begin(), i.rollingSpring.end());
		torsionSpring.insert(torsionSpring.end(), i.torsionSpring.begin(), i.torsionSpring.end());
		objectPointed.insert(objectPointed.end(), i.objectPointed.begin(), i.objectPointed.end());
		objectPointing.insert(objectPointing.end(), i.objectPointing.begin(), i.objectPointing.end());
		capacity += i.capacity;
	}
};

struct HostInteractionBonded
{
	int num{ 0 };
	std::vector<double3> contactNormal;
	std::vector<double3> contactPoint;
	std::vector<double3> shearForce;
	std::vector<double3> bendingTorque;
	std::vector<double> normalForce;
	std::vector<double> torsionTorque;
	std::vector<int> objectPointed;
	std::vector<int> objectPointing;
	std::vector<int> isBonded;

	HostInteractionBonded() = default;

	explicit HostInteractionBonded(int n)
		:num(n), contactNormal(n, make_double3(0, 0, 0)), contactPoint(n, make_double3(0, 0, 0)), shearForce(n, make_double3(0, 0, 0)), bendingTorque(n, make_double3(0, 0, 0)), normalForce(n, 0.), torsionTorque(n, 0.), objectPointed(n, -1), objectPointing(n, -1), isBonded(n, 0)
	{
	}

	void insertData(const HostInteractionBonded i)
	{
		contactNormal.insert(contactNormal.end(), i.contactNormal.begin(), i.contactNormal.end());
		contactPoint.insert(contactPoint.end(), i.contactPoint.begin(), i.contactPoint.end());
		shearForce.insert(shearForce.end(), i.shearForce.begin(), i.shearForce.end());
		bendingTorque.insert(bendingTorque.end(), i.bendingTorque.begin(), i.bendingTorque.end());
		normalForce.insert(normalForce.end(), i.normalForce.begin(), i.normalForce.end());
		torsionTorque.insert(torsionTorque.end(), i.torsionTorque.begin(), i.torsionTorque.end());
		objectPointed.insert(objectPointed.end(), i.objectPointed.begin(), i.objectPointed.end());
		objectPointing.insert(objectPointing.end(), i.objectPointing.begin(), i.objectPointing.end());
		isBonded.insert(isBonded.end(), i.isBonded.begin(), i.isBonded.end());
		num += i.num;
	}
};

struct HostHertzianContactModel
{
	std::vector<double> E;
	std::vector<double> G;
	std::vector<double> res;
	std::vector<double> k_r_k_s;
	std::vector<double> k_t_k_s;
	std::vector<double> mu_s;
	std::vector<double> mu_r;
	std::vector<double> mu_t;

	HostHertzianContactModel() = default;

	explicit HostHertzianContactModel(int n)
		: E(n, 0), G(n, 0), res(n, 1.), k_r_k_s(n, 0), k_t_k_s(n, 0), mu_s(n, 0), mu_r(n, 0), mu_t(n, 0)
	{
	}
};

struct HostLinearContactModel
{
	std::vector<double> k_n;
	std::vector<double> k_s;
	std::vector<double> k_r;
	std::vector<double> k_t;
	std::vector<double> d_n;
	std::vector<double> d_s;
	std::vector<double> d_r;
	std::vector<double> d_t;
	std::vector<double> mu_s;
	std::vector<double> mu_r;
	std::vector<double> mu_t;
	//k: stiffness, d: dissipation, mu: frictionCoefficient
	//n: normal, s: shear, r: rolling, t: torsion

	HostLinearContactModel() = default;

	explicit HostLinearContactModel(int n)
		: k_n(n, 0), k_s(n, 0), k_r(n, 0), k_t(n, 0), d_n(n, 0), d_s(n, 0), d_r(n, 0), d_t(n, 0), mu_s(n, 0), mu_r(n, 0), mu_t(n, 0)
	{
	}
};

struct HostBondedContactModel
{
	std::vector<double> gamma;//multiplier
	std::vector<double> E;
	std::vector<double> k_n_k_s;
	std::vector<double> sigma_s;//tensile strength
	std::vector<double> C;//cohesion
	std::vector<double> mu;//friction coefficient

	HostBondedContactModel() = default;

	explicit HostBondedContactModel(int n)
		: gamma(n, 1), E(n, 0), k_n_k_s(n, 2), sigma_s(n, 0), C(n, 0), mu(n, 0)
	{
	}
};

struct HostSolidContactModel
{
	int nMaterial{ 0 };//number of materials
	HostHertzianContactModel hertzian;
	HostLinearContactModel linear;
	HostBondedContactModel bonded;

	HostSolidContactModel() = default;

	explicit HostSolidContactModel(int nMat)
	{
		nMaterial = nMat;
		int nCombined = nMat * (nMat + 1) / 2;
		hertzian = HostHertzianContactModel(nCombined);
		linear = HostLinearContactModel(nCombined);
		bonded = HostBondedContactModel(nCombined);
	}

	int getCombinedIndex(int mA, int mB) const
	{
		if (mA >= nMaterial || mB >= nMaterial)
		{
			return -1;
		}
		int i = int(mA);
		int j = int(mB);
		if (mA > mB)
		{
			i = int(mB);
			j = int(mA);
		}
		int index = (i * (2 * nMaterial - i + 1)) / 2 + j - i;
		return index;
	}
};

struct HostData
{
	HostFluid fluids;
	HostSolid solids;
	HostClump clumps;
	HostInteractionBase fluid2Fluid;
	HostInteractionBase fluid2Solid;
	HostInteractionSolid2Solid solid2Solid;
	HostInteractionBonded solidBond2Solid;
	HostSolidContactModel contactModels;
};