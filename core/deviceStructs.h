#pragma once
#include "myCUDA.h"
#include "hostStructs.h"

struct objectHash 
{
	int* value{ nullptr };
	int* aux{ nullptr };
	int* index{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(value, n, InitMode::NEG_ONE);
		CUDA_ALLOC(aux, n, InitMode::NEG_ONE);
		CUDA_ALLOC(index, n, InitMode::NEG_ONE);
	}

	void release()
	{
		CUDA_FREE(value);
		CUDA_FREE(aux);
		CUDA_FREE(index);
	}

	void reset(int n) const
	{
		CUDA_CHECK(cudaMemset(value, 0xFFFFFFFF, n * sizeof(int)));
		CUDA_CHECK(cudaMemset(aux, 0xFFFFFFFF, n * sizeof(int)));
		CUDA_CHECK(cudaMemset(index, 0xFFFFFFFF, n * sizeof(int)));
	}
};

struct objectNeighborPrefix
{
	int* count{ nullptr };
	int* prefixSum{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(count, n, InitMode::ZERO);
		CUDA_ALLOC(prefixSum, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(count);
		CUDA_FREE(prefixSum);
	}
};

struct interactionRange 
{
	int* start{ nullptr };
	int* end{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(start, n, InitMode::NEG_ONE);
		CUDA_ALLOC(end, n, InitMode::NEG_ONE);
	}

	void release()
	{
		CUDA_FREE(start);
		CUDA_FREE(end);
	}

	void reset(int n) const
	{
		CUDA_CHECK(cudaMemset(start, 0xFFFFFFFF, n * sizeof(int)));
		CUDA_CHECK(cudaMemset(end, 0xFFFFFFFF, n * sizeof(int)));
	}
};

struct pointCloud
{
	int num{ 0 };
	objectHash hash;
	double3* position{ nullptr };
	double* effectiveRadii{ nullptr };

	void alloc(int n)
	{
		num = n;
		hash.alloc(n);
		CUDA_ALLOC(position, n, InitMode::ZERO);
		CUDA_ALLOC(effectiveRadii, n, InitMode::ZERO);
	}

	void release()
	{
		hash.release();
		CUDA_FREE(position);
		CUDA_FREE(effectiveRadii);
		num = 0;
	}

	void download(const HostPointCloud& p)
	{
		int nDownload = p.num;
		cuda_copy(position, p.position.data(), nDownload, CopyDir::H2D);
		cuda_copy(effectiveRadii, p.effectiveRadii.data(), nDownload, CopyDir::H2D);
	}

	void uploadState(HostPointCloud& p) const
	{
		int nUpload = num;
		cuda_copy(p.position.data(), position, nUpload, CopyDir::D2H);
	}
};

struct dynamicStateBase
{
	double3* velocities{ nullptr };
	double3* accelerations{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(velocities, n, InitMode::ZERO);
		CUDA_ALLOC(accelerations, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(velocities);
		CUDA_FREE(accelerations);
	}

	void upload(int n, HostDynamicStateBase& h) const
	{
		cuda_copy(h.velocities.data(), velocities, n, CopyDir::D2H);
		cuda_copy(h.accelerations.data(), accelerations, n, CopyDir::D2H);
	}

	void download(int n, const HostDynamicStateBase& h) const
	{
		cuda_copy(velocities, h.velocities.data(), n, CopyDir::H2D);
		cuda_copy(accelerations, h.accelerations.data(), n, CopyDir::H2D);
	}
};

struct fluid
{
	pointCloud points;
	dynamicStateBase dyn;
	objectNeighborPrefix fluidNeighbor;
	objectNeighborPrefix solidNeighbor;
	interactionRange fluidRange;
	double* rho0{ nullptr }; // initial density
	double* dRho{ nullptr }; // density change
	double* c{ nullptr }; // speed of sound
	double* v{ nullptr }; // kinematic viscosity
	double* sumW0{ nullptr };

	void alloc(int n)
	{
		points.alloc(n);
		dyn.alloc(n);
		fluidNeighbor.alloc(n);
		solidNeighbor.alloc(n);
		fluidRange.alloc(n);
		CUDA_ALLOC(rho0, n, InitMode::ZERO);
		CUDA_ALLOC(dRho, n, InitMode::ZERO);
		CUDA_ALLOC(c, n, InitMode::ZERO);
		CUDA_ALLOC(v, n, InitMode::ZERO);
		CUDA_ALLOC(sumW0, n, InitMode::ZERO);
	}

	void release()
	{
		points.release();
		dyn.release();
		fluidNeighbor.release();
		solidNeighbor.release();
		fluidRange.release();
		CUDA_FREE(rho0);
		CUDA_FREE(dRho);
		CUDA_FREE(c);
		CUDA_FREE(v);
		CUDA_FREE(sumW0);
	}

	void copy(const HostFluid& f)
	{
		release();
		int nCopy = f.points.num;
		alloc(nCopy);
		points.download(f.points);
		dyn.download(nCopy, f.dyn);
		cuda_copy(rho0, f.rho0.data(), nCopy, CopyDir::H2D);
		cuda_copy(dRho, f.dRho.data(), nCopy, CopyDir::H2D);
		cuda_copy(c, f.c.data(), nCopy, CopyDir::H2D);
		cuda_copy(v, f.v.data(), nCopy, CopyDir::H2D);
	}

	void uploadState(HostFluid& f) const
	{
		int nUpload = points.num;
		points.uploadState(f.points);
		dyn.upload(nUpload, f.dyn);
		cuda_copy(f.dRho.data(), dRho, nUpload, CopyDir::D2H);
	}
};

struct solid
{
	pointCloud points;
	dynamicStateBase dyn;
	objectNeighborPrefix solidNeighbor;
	interactionRange solidRange;
	interactionRange fluidSolidRange;
	double3* normal{ nullptr };
	double3* torques{ nullptr };
	double3* angularVelocities{ nullptr };
	double* radius{ nullptr };
	double* inverseMass{ nullptr };
	int* materialID{ nullptr };
	int* clusterID{ nullptr };
	int* clumpID{ nullptr };

	double* pressure{ nullptr };
	double3* smoothedVelocity{ nullptr };

	void alloc(int n)
	{
		points.alloc(n);
		dyn.alloc(n);
		solidNeighbor.alloc(n);
		solidRange.alloc(n);
		fluidSolidRange.alloc(n);
		CUDA_ALLOC(normal, n, InitMode::ZERO);
		CUDA_ALLOC(torques, n, InitMode::ZERO);
		CUDA_ALLOC(angularVelocities, n, InitMode::ZERO);
		CUDA_ALLOC(radius, n, InitMode::ZERO);
		CUDA_ALLOC(inverseMass, n, InitMode::ZERO);
		CUDA_ALLOC(materialID, n, InitMode::ZERO);
		CUDA_ALLOC(clusterID, n, InitMode::NEG_ONE);
		CUDA_ALLOC(clumpID, n, InitMode::NEG_ONE);

		CUDA_ALLOC(pressure, n, InitMode::ZERO);
		CUDA_ALLOC(smoothedVelocity, n, InitMode::ZERO);
	}

	void release()
	{
		points.release();
		dyn.release();
		solidNeighbor.release();
		solidRange.release();
		fluidSolidRange.release();
		CUDA_FREE(normal);
		CUDA_FREE(torques);
		CUDA_FREE(angularVelocities);
		CUDA_FREE(radius);
		CUDA_FREE(inverseMass);
		CUDA_FREE(materialID);
		CUDA_FREE(clusterID);
		CUDA_FREE(clumpID);

		CUDA_FREE(pressure);
		CUDA_FREE(smoothedVelocity);
	}

	void copy(const HostSolid& s)
	{
		release();
		int nCopy = s.points.num;
		alloc(nCopy);
		points.download(s.points);
		dyn.download(nCopy, s.dyn);
		cuda_copy(normal, s.normal.data(), nCopy, CopyDir::H2D);
		cuda_copy(torques, s.torques.data(), nCopy, CopyDir::H2D);
		cuda_copy(angularVelocities, s.angularVelocities.data(), nCopy, CopyDir::H2D);
		cuda_copy(radius, s.radius.data(), nCopy, CopyDir::H2D);
		cuda_copy(inverseMass, s.inverseMass.data(), nCopy, CopyDir::H2D);
		cuda_copy(materialID, s.materialID.data(), nCopy, CopyDir::H2D);
		cuda_copy(clusterID, s.clusterID.data(), nCopy, CopyDir::H2D);
		cuda_copy(clumpID, s.clumpID.data(), nCopy, CopyDir::H2D);

		cuda_copy(pressure, s.pressure.data(), nCopy, CopyDir::H2D);
	}

	void uploadState(HostSolid& s) const
	{
		int nUpload = points.num;
		points.uploadState(s.points);
		dyn.upload(nUpload, s.dyn);
		cuda_copy(s.normal.data(), normal, nUpload, CopyDir::D2H);
		cuda_copy(s.torques.data(), torques, nUpload, CopyDir::D2H);
		cuda_copy(s.angularVelocities.data(), angularVelocities, nUpload, CopyDir::D2H);
		cuda_copy(s.pressure.data(), pressure, nUpload, CopyDir::D2H);
	}
};

struct clump
{
	int num{ 0 };
	dynamicStateBase dyn;
	double3* centroidPosition{ nullptr };
	double3* torques{ nullptr };
	double3* angularVelocities{ nullptr };
	quaternion* orientations{ nullptr };
	symMatrix* inverseInertiaTensor{ nullptr };
	double* inverseMass{ nullptr };
	int* pebbleStartIndex{ nullptr };
	int* pebbleEndIndex{ nullptr };

	void alloc(int n)
	{
		num = n;
		dyn.alloc(n);
		CUDA_ALLOC(centroidPosition, n, InitMode::ZERO);
		CUDA_ALLOC(torques, n, InitMode::ZERO);
		CUDA_ALLOC(angularVelocities, n, InitMode::ZERO);
		CUDA_ALLOC(orientations, n, InitMode::ZERO);
		CUDA_ALLOC(inverseInertiaTensor, n, InitMode::ZERO);
		CUDA_ALLOC(inverseMass, n, InitMode::ZERO);
		CUDA_ALLOC(pebbleStartIndex, n, InitMode::NEG_ONE);
		CUDA_ALLOC(pebbleEndIndex, n, InitMode::NEG_ONE);
	}

	void release()
	{
		dyn.release();
		CUDA_FREE(centroidPosition);
		CUDA_FREE(torques);
		CUDA_FREE(angularVelocities);
		CUDA_FREE(orientations);
		CUDA_FREE(inverseInertiaTensor);
		CUDA_FREE(inverseMass);
		CUDA_FREE(pebbleStartIndex);
		CUDA_FREE(pebbleEndIndex);
		num = 0;
	}

	void copy(const HostClump& c)
	{
		release();
		int nCopy = c.num;
		alloc(nCopy);
		dyn.download(nCopy, c.dyn);
		cuda_copy(centroidPosition, c.centroidPosition.data(), nCopy, CopyDir::H2D);
		cuda_copy(torques, c.torques.data(), nCopy, CopyDir::H2D);
		cuda_copy(angularVelocities, c.angularVelocities.data(), nCopy, CopyDir::H2D);
		cuda_copy(orientations, c.orientations.data(), nCopy, CopyDir::H2D);
		cuda_copy(inverseInertiaTensor, c.inverseInertiaTensor.data(), nCopy, CopyDir::H2D);
		cuda_copy(inverseMass, c.inverseMass.data(), nCopy, CopyDir::H2D);
		cuda_copy(pebbleStartIndex, c.pebbleStartIndex.data(), nCopy, CopyDir::H2D);
		cuda_copy(pebbleEndIndex, c.pebbleEndIndex.data(), nCopy, CopyDir::H2D);
	}

	void uploadState(HostClump& c) const
	{
		int nUpload = num;
		dyn.upload(nUpload, c.dyn);
		cuda_copy(c.centroidPosition.data(), centroidPosition, nUpload, CopyDir::D2H);
		cuda_copy(c.torques.data(), torques, nUpload, CopyDir::D2H);
		cuda_copy(c.angularVelocities.data(), angularVelocities, nUpload, CopyDir::D2H);
		cuda_copy(c.orientations.data(), orientations, nUpload, CopyDir::D2H);
	}
};

struct interactionBase
{
	int capacity{ 0 }, num{ 0 };
	objectHash hash;
	double3* force{ nullptr };
	int* objectPointed{ nullptr };
	int* objectPointing{ nullptr };

	void alloc(int n)
	{
		capacity = n;
		num = 0;
		hash.alloc(n);
		CUDA_ALLOC(force, n, InitMode::ZERO);
		CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE);
		CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE);
	}

	void release()
	{
		capacity = 0;
		num = 0;
		hash.release();
		CUDA_FREE(force);
		CUDA_FREE(objectPointed);
		CUDA_FREE(objectPointing);
	}

	void copy(const HostInteractionBase& i)
	{
		release();
		int nCopy = i.capacity;
		alloc(nCopy);
		cuda_copy(force, i.force.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointed, i.objectPointed.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointing, i.objectPointing.data(), nCopy, CopyDir::H2D);
		num = i.num;
	}

	void setNum(int n)
	{
		if (n > capacity)
		{
			release();
			alloc(n);
		}
		num = n;
	}

	void upload(HostInteractionBase& h) const
	{
		int nUpload = num;
		if (nUpload > h.capacity)
		{
			h = HostInteractionBase(nUpload);
		}
		cuda_copy(h.force.data(), force, nUpload, CopyDir::D2H);
		cuda_copy(h.objectPointed.data(), objectPointed, nUpload, CopyDir::D2H);
		cuda_copy(h.objectPointing.data(), objectPointing, nUpload, CopyDir::D2H);
		h.num = num;
	}

	void setHash() const
	{
		cuda_copy(hash.value, objectPointing, num, CopyDir::D2D);
	}
};

struct interactionHistory
{
	int capacity{ 0 }, num{ 0 };
	int* objectPointed{ nullptr };
	int* objectPointing{ nullptr };
	double3* contactNormal{ nullptr };
	double3* slidingSpring{ nullptr };
	double3* rollingSpring{ nullptr };
	double3* torsionSpring{ nullptr };

	void alloc(int n)
	{
		capacity = n;
		num = 0;
		CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE);
		CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE);
		CUDA_ALLOC(contactNormal, n, InitMode::ZERO);
		CUDA_ALLOC(slidingSpring, n, InitMode::ZERO);
		CUDA_ALLOC(rollingSpring, n, InitMode::ZERO);
		CUDA_ALLOC(torsionSpring, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(objectPointed);
		CUDA_FREE(objectPointing);
		CUDA_FREE(contactNormal);
		CUDA_FREE(slidingSpring);
		CUDA_FREE(rollingSpring);
		CUDA_FREE(torsionSpring);
		capacity = 0;
		num = 0;
	}
};

struct interactionSolid2Solid
{
	int capacity{ 0 }, num{ 0 };
	objectHash hash;
	interactionHistory history;
	double3* force{ nullptr };
	double3* torque{ nullptr };
	double3* slidingSpring{ nullptr };
	double3* rollingSpring{ nullptr };
	double3* torsionSpring{ nullptr };
	int* objectPointed{ nullptr };
	int* objectPointing{ nullptr };

	void alloc(int n)
	{
		capacity = n;
		num = 0;
		hash.alloc(n);
		history.alloc(n);
		CUDA_ALLOC(force, n, InitMode::ZERO);
		CUDA_ALLOC(torque, n, InitMode::ZERO);
		CUDA_ALLOC(slidingSpring, n, InitMode::ZERO);
		CUDA_ALLOC(rollingSpring, n, InitMode::ZERO);
		CUDA_ALLOC(torsionSpring, n, InitMode::ZERO);
		CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE);
		CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE);
	}

	void release()
	{
		hash.release();
		history.release();
		CUDA_FREE(force);
		CUDA_FREE(torque);
		CUDA_FREE(slidingSpring);
		CUDA_FREE(rollingSpring);
		CUDA_FREE(torsionSpring);
		CUDA_FREE(objectPointed);
		CUDA_FREE(objectPointing);
	}

	void copy(const HostInteractionSolid2Solid& i)
	{
		release();
		int nCopy = i.capacity;
		alloc(nCopy);
		cuda_copy(force, i.force.data(), nCopy, CopyDir::H2D);
		cuda_copy(torque, i.torque.data(), nCopy, CopyDir::H2D);
		cuda_copy(slidingSpring, i.slidingSpring.data(), nCopy, CopyDir::H2D);
		cuda_copy(rollingSpring, i.rollingSpring.data(), nCopy, CopyDir::H2D);
		cuda_copy(torsionSpring, i.torsionSpring.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointed, i.objectPointed.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointing, i.objectPointing.data(), nCopy, CopyDir::H2D);
		num = i.num;
	}

	void upload(HostInteractionSolid2Solid& i) const
	{
		int nUpload = num;
		if (nUpload > i.capacity)
		{
			i = HostInteractionSolid2Solid(nUpload);
		}
		cuda_copy(i.force.data(), force, nUpload, CopyDir::D2H);
		cuda_copy(i.torque.data(), torque, nUpload, CopyDir::D2H);
		cuda_copy(i.slidingSpring.data(), slidingSpring, nUpload, CopyDir::D2H);
		cuda_copy(i.rollingSpring.data(), rollingSpring, nUpload, CopyDir::D2H);
		cuda_copy(i.torsionSpring.data(), torsionSpring, nUpload, CopyDir::D2H);
		cuda_copy(i.objectPointed.data(), objectPointed, nUpload, CopyDir::D2H);
		cuda_copy(i.objectPointing.data(), objectPointing, nUpload, CopyDir::D2H);
		i.num = num;
	}

	void setNum(int n)
	{
		if (n > capacity)
		{
			hash.release();
			CUDA_FREE(force);
			CUDA_FREE(torque);
			CUDA_FREE(slidingSpring);
			CUDA_FREE(rollingSpring);
			CUDA_FREE(torsionSpring);
			CUDA_FREE(objectPointed);
			CUDA_FREE(objectPointing);			
			capacity = n;
			hash.alloc(n);
			CUDA_ALLOC(force, n, InitMode::ZERO);
			CUDA_ALLOC(torque, n, InitMode::ZERO);
			CUDA_ALLOC(slidingSpring, n, InitMode::ZERO);
			CUDA_ALLOC(rollingSpring, n, InitMode::ZERO);
			CUDA_ALLOC(torsionSpring, n, InitMode::ZERO);
			CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE);
			CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE);
		}
		num = n;
	}

	void save()
	{
		if (num > history.capacity)
		{
			history.release();
			history.alloc(num);
		}
		history.num = num;
		cuda_copy(history.objectPointed, objectPointed, num, CopyDir::D2D);
		cuda_copy(history.objectPointing, objectPointing, num, CopyDir::D2D);
		cuda_copy(history.slidingSpring, slidingSpring, num, CopyDir::D2D);
		cuda_copy(history.rollingSpring, rollingSpring, num, CopyDir::D2D);
		cuda_copy(history.torsionSpring, torsionSpring, num, CopyDir::D2D);
	}

	void setHash() const
	{
		cuda_copy(hash.value, objectPointing, num, CopyDir::D2D);
	}
};

struct interactionBonded
{
	int num{ 0 };
	double3* contactNormal{ nullptr };
	double3* contactPoint{ nullptr };
	double3* shearForce{ nullptr };
	double3* bendingTorque{ nullptr };
	double* normalForce{ nullptr };
	double* torsionTorque{ nullptr };
	int* objectPointed{ nullptr };
	int* objectPointing{ nullptr };
	int* isBonded{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(contactNormal, n, InitMode::ZERO);
		CUDA_ALLOC(contactPoint, n, InitMode::ZERO);
		CUDA_ALLOC(shearForce, n, InitMode::ZERO);
		CUDA_ALLOC(bendingTorque, n, InitMode::ZERO);
		CUDA_ALLOC(normalForce, n, InitMode::ZERO);
		CUDA_ALLOC(torsionTorque, n, InitMode::ZERO);
		CUDA_ALLOC(objectPointed, n, InitMode::NEG_ONE);
		CUDA_ALLOC(objectPointing, n, InitMode::NEG_ONE);
		CUDA_ALLOC(isBonded, n, InitMode::ZERO);
	}

	void release()
	{
		num = 0;
		CUDA_FREE(contactNormal);
		CUDA_FREE(contactPoint);
		CUDA_FREE(shearForce);
		CUDA_FREE(bendingTorque);
		CUDA_FREE(normalForce);
		CUDA_FREE(torsionTorque);
		CUDA_FREE(objectPointed);
		CUDA_FREE(objectPointing);
		CUDA_FREE(isBonded);
	}

	void copy(const HostInteractionBonded& i)
	{
		release();
		int nCopy = i.num;
		alloc(nCopy);
		cuda_copy(contactNormal, i.contactNormal.data(), nCopy, CopyDir::H2D);
		cuda_copy(contactPoint, i.contactPoint.data(), nCopy, CopyDir::H2D);
		cuda_copy(shearForce, i.shearForce.data(), nCopy, CopyDir::H2D);
		cuda_copy(bendingTorque, i.bendingTorque.data(), nCopy, CopyDir::H2D);
		cuda_copy(normalForce, i.normalForce.data(), nCopy, CopyDir::H2D);
		cuda_copy(torsionTorque, i.torsionTorque.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointed, i.objectPointed.data(), nCopy, CopyDir::H2D);
		cuda_copy(objectPointing, i.objectPointing.data(), nCopy, CopyDir::H2D);
		cuda_copy(isBonded, i.isBonded.data(), nCopy, CopyDir::H2D);
		num = i.num;
	}

	void upload(HostInteractionBonded& i) const
	{
		int nUpload = num;
		if (nUpload > i.num)
		{
			i = HostInteractionBonded(nUpload);
		}
		cuda_copy(i.contactNormal.data(), contactNormal, nUpload, CopyDir::D2H);
		cuda_copy(i.contactPoint.data(), contactPoint, nUpload, CopyDir::D2H);
		cuda_copy(i.shearForce.data(), shearForce, nUpload, CopyDir::D2H);
		cuda_copy(i.bendingTorque.data(), bendingTorque, nUpload, CopyDir::D2H);
		cuda_copy(i.normalForce.data(), normalForce, nUpload, CopyDir::D2H);
		cuda_copy(i.torsionTorque.data(), torsionTorque, nUpload, CopyDir::D2H);
		cuda_copy(i.objectPointed.data(), objectPointed, nUpload, CopyDir::D2H);
		cuda_copy(i.objectPointing.data(), objectPointing, nUpload, CopyDir::D2H);
		cuda_copy(i.isBonded.data(), isBonded, nUpload, CopyDir::D2H);
		i.num = num;
	}
};

struct hertzianContactModel
{
	double* E{ nullptr };
	double* G{ nullptr };
	double* res{ nullptr };
	double* k_r_k_s{ nullptr };
	double* k_t_k_s{ nullptr };
	double* mu_s{ nullptr };
	double* mu_r{ nullptr };
	double* mu_t{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(E, n, InitMode::ZERO);
		CUDA_ALLOC(G, n, InitMode::ZERO);
		CUDA_ALLOC(res, n, InitMode::ZERO);
		CUDA_ALLOC(k_r_k_s, n, InitMode::ZERO);
		CUDA_ALLOC(k_t_k_s, n, InitMode::ZERO);
		CUDA_ALLOC(mu_s, n, InitMode::ZERO);
		CUDA_ALLOC(mu_r, n, InitMode::ZERO);
		CUDA_ALLOC(mu_t, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(E);
		CUDA_FREE(G);
		CUDA_FREE(res);
		CUDA_FREE(k_r_k_s);
		CUDA_FREE(k_t_k_s);
		CUDA_FREE(mu_s);
		CUDA_FREE(mu_r);
		CUDA_FREE(mu_t);
	}
};

struct linearContactModel
{
	double* k_n{ nullptr };
	double* k_s{ nullptr };
	double* k_r{ nullptr };
	double* k_t{ nullptr };
	double* d_n{ nullptr };
	double* d_s{ nullptr };
	double* d_r{ nullptr };
	double* d_t{ nullptr };
	double* mu_s{ nullptr };
	double* mu_r{ nullptr };
	double* mu_t{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(k_n, n, InitMode::ZERO);
		CUDA_ALLOC(k_s, n, InitMode::ZERO);
		CUDA_ALLOC(k_r, n, InitMode::ZERO);
		CUDA_ALLOC(k_t, n, InitMode::ZERO);
		CUDA_ALLOC(d_n, n, InitMode::ZERO);
		CUDA_ALLOC(d_s, n, InitMode::ZERO);
		CUDA_ALLOC(d_r, n, InitMode::ZERO);
		CUDA_ALLOC(d_t, n, InitMode::ZERO);
		CUDA_ALLOC(mu_s, n, InitMode::ZERO);
		CUDA_ALLOC(mu_r, n, InitMode::ZERO);
		CUDA_ALLOC(mu_t, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(k_n);
		CUDA_FREE(k_s);
		CUDA_FREE(k_r);
		CUDA_FREE(k_t);
		CUDA_FREE(d_n);
		CUDA_FREE(d_s);
		CUDA_FREE(d_r);
		CUDA_FREE(d_t);
		CUDA_FREE(mu_s);
		CUDA_FREE(mu_r);
		CUDA_FREE(mu_t);
	}
};

struct bondedContactModel
{
	double* gamma{ nullptr };
	double* E{ nullptr };
	double* k_n_k_s{ nullptr };
	double* sigma_s{ nullptr };
	double* C{ nullptr };
	double* mu{ nullptr };

	void alloc(int n)
	{
		CUDA_ALLOC(gamma, n, InitMode::ZERO);
		CUDA_ALLOC(E, n, InitMode::ZERO);
		CUDA_ALLOC(k_n_k_s, n, InitMode::ZERO);
		CUDA_ALLOC(sigma_s, n, InitMode::ZERO);
		CUDA_ALLOC(C, n, InitMode::ZERO);
		CUDA_ALLOC(mu, n, InitMode::ZERO);
	}

	void release()
	{
		CUDA_FREE(gamma);
		CUDA_FREE(E);
		CUDA_FREE(k_n_k_s);
		CUDA_FREE(sigma_s);
		CUDA_FREE(C);
		CUDA_FREE(mu);
	}
};

struct solidContactModel
{
	int nMaterial{0};
	hertzianContactModel hertzian;
	linearContactModel linear;
	bondedContactModel bonded;

	void alloc(int n)
	{
		hertzian.alloc(n);
		linear.alloc(n);
		bonded.alloc(n);
	}

	void release()
	{
		hertzian.release();
		linear.release();
		bonded.release();
	}

	void copy(HostSolidContactModel s)
	{
		release();
		nMaterial = s.nMaterial;
		int nCombined = nMaterial * (nMaterial + 1) / 2;
		int nCopy = int(s.hertzian.E.size());
		if (nCopy >= nCombined) alloc(nCopy);
		else alloc(nCombined);
		cuda_copy(hertzian.E, s.hertzian.E.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.G, s.hertzian.G.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.res, s.hertzian.res.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.k_r_k_s, s.hertzian.k_r_k_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.k_t_k_s, s.hertzian.k_t_k_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.mu_s, s.hertzian.mu_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.mu_r, s.hertzian.mu_r.data(), nCopy, CopyDir::H2D);
		cuda_copy(hertzian.mu_t, s.hertzian.mu_t.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.k_n, s.linear.k_n.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.k_s, s.linear.k_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.k_r, s.linear.k_r.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.k_t, s.linear.k_t.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.d_n, s.linear.d_n.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.d_s, s.linear.d_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.d_r, s.linear.d_r.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.d_t, s.linear.d_t.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.mu_s, s.linear.mu_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.mu_r, s.linear.mu_r.data(), nCopy, CopyDir::H2D);
		cuda_copy(linear.mu_t, s.linear.mu_t.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.gamma, s.bonded.gamma.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.E, s.bonded.E.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.k_n_k_s, s.bonded.k_n_k_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.sigma_s, s.bonded.sigma_s.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.C, s.bonded.C.data(), nCopy, CopyDir::H2D);
		cuda_copy(bonded.mu, s.bonded.mu.data(), nCopy, CopyDir::H2D);
	}

	int __device__ getCombinedIndex(int mA, int mB) const
	{
		if (mA >= nMaterial || mB >= nMaterial)
		{
			return -1;
		}
		int i = mA;
		int j = mB;
		if (mA > mB)
		{
			i = mB;
			j = mA;
		}
		int index = (i * (2 * nMaterial - i + 1)) / 2 + j - i;
		return index;
	}
};

struct spatialGrid 
{
	int      num{ 0 };
	double3  minBound{ make_double3(0., 0., 0.) };
	double3  maxBound{ make_double3(1.,1.,1.) };
	double3  cellSize{ make_double3(1.,1.,1.) };
	int3     gridSize{ make_int3(1, 1, 1) }; // x * y * z + 1 = num

	int* cellStartFluid{ nullptr };
	int* cellEndFluid{ nullptr };
	int* cellStartSolid{ nullptr };
	int* cellEndSolid{ nullptr };

	void alloc(int nCell)
	{
		num = nCell;
		CUDA_ALLOC(cellStartFluid, nCell, InitMode::NEG_ONE);
		CUDA_ALLOC(cellEndFluid, nCell, InitMode::NEG_ONE);
		CUDA_ALLOC(cellStartSolid, nCell, InitMode::NEG_ONE);
		CUDA_ALLOC(cellEndSolid, nCell, InitMode::NEG_ONE);
	}

	void release()
	{
		CUDA_FREE(cellStartFluid);
		CUDA_FREE(cellEndFluid);
		CUDA_FREE(cellStartSolid);
		CUDA_FREE(cellEndSolid);
		num = 0;
	}

	void resetCellStartEnd() const
	{
		CUDA_CHECK(cudaMemset(cellStartFluid, 0xFFFFFFFF, num * sizeof(int)));
		CUDA_CHECK(cudaMemset(cellEndFluid, 0xFFFFFFFF, num * sizeof(int)));
		CUDA_CHECK(cudaMemset(cellStartSolid, 0xFFFFFFFF, num * sizeof(int)));
		CUDA_CHECK(cudaMemset(cellEndSolid, 0xFFFFFFFF, num * sizeof(int)));
	}
};

struct DeviceData
{
	double3 gravity{ make_double3(0, 0, -9.81) };
	spatialGrid spatialGrids;
	fluid fluids;
	solid solids;
	clump clumps;
	interactionBase fluid2Fluid;
	interactionBase fluid2Solid;
	interactionSolid2Solid solid2Solid;
	interactionBonded solidBond2Solid;
	solidContactModel contactModels;

	void release()
	{
		spatialGrids.release();
		fluids.release();
		solids.release();
		clumps.release();
		fluid2Fluid.release();
		fluid2Solid.release();
		solid2Solid.release();
		solidBond2Solid.release();
		contactModels.release();
	}
};