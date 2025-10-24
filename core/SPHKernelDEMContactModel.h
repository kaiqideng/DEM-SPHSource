#pragma once
#include "neighborSearch.h"

// Wendland 5th-order (C2) kernel in 3D
__device__ __forceinline__ double wendlandKernel3D(double r, double h)
{
	double q = r / h;
	if (q >= 2.0) return 0.0;

	double sigma = 21.0 / (16.0 * pi() * h * h * h);
	double term = 1.0 - 0.5 * q;
	return sigma * pow(term, 4) * (1.0 + 2.0 * q);
}

__device__ __forceinline__ double3 gradWendlandKernel3D(const double3& rij, double h)
{
	double r = length(rij);
	if (r < 1.e-10 || r >= 2.0 * h) return make_double3(0, 0, 0);

	double q = r / h;
	double sigma = 21.0 / (16.0 * pi() * h * h * h);

	double term = 1.0 - 0.5 * q;
	// d/dq of [ (1 - q/2)^4 (1 + 2q) ]
	double dW_dq = (-2.0 * pow(term, 3) * (1.0 + 2.0 * q) + 2.0 * pow(term, 4));

	// chain rule: dW/dr = (dW/dq) * (dq/dr) = (dW/dq) / h
	double dWdr = sigma * dW_dq / h;

	double factor = dWdr / r;  // multiply with rij/r
	return factor * rij;
}

__device__ __forceinline__ double getPressure(double rho0, double dRho, double c, double chi)
{
	double rho = rho0 + dRho;
	double p0 = c * c * rho0 / 7.;
	return p0 * (pow(rho / rho0, 7.) - 1.) + chi;
}

__device__ __forceinline__ double getDensityFromPressure(double pressure, double rho0, double c, double chi)
{
	double p0 = c * c * rho0 / 7.;
	return rho0 * pow((pressure - chi) / p0 + 1., 1. / 7.);
}

__device__ __forceinline__ double getDensityIncrementFromParticleJ(double rho_i, double V_j, double3 v_ij, double3 gradW_ij)
{
	return rho_i * V_j * dot(v_ij, gradW_ij);
}

__device__ __forceinline__ double getAveragePressure(double P_i, double rho_i, double P_j, double rho_j)
{
	return (rho_i * P_j + rho_j * P_i) / (rho_i + rho_j);
}

__device__ __forceinline__ double3 getPressureAccelerationFromParticleJ(double m_i, double V_i, double V_j, double P_ij, double3 gradW_ij)
{
	return -1. / m_i * (V_i * V_i + V_j * V_j) * P_ij * gradW_ij;
}

__device__ __forceinline__ double getCombinedViscosity(double rho_i, double nu_i, double rho_j, double nu_j)
{
	double gamma_i = rho_i * nu_i;
	double gamma_j = rho_j * nu_j;
	return 2. * gamma_i * gamma_j / (gamma_i + gamma_j);
}

__device__ __forceinline__ double3 getViscosityAccelerationFromParticleJ(double m_i, double V_i, double V_j, double gamma_ij, double3 v_ij, double3 r_ij, double3 gradW_ij)
{
	return 1. / m_i * gamma_ij * (V_i * V_i + V_j * V_j) * v_ij / length(r_ij) * dot(normalize(r_ij), gradW_ij);
}

__device__ __forceinline__ double3 getArtificialViscosityAccelerationFromParticleJ(double m_j, double h_ij, double c_ij, double rho_ij, double alpha, double3 r_ij, double3 v_ij, double3 gradW_ij)//only for fluid particles
{
	return -m_j * alpha * h_ij * c_ij * dot(v_ij, r_ij) / (rho_ij * lengthSquared(r_ij) + 0.01 * h_ij * h_ij) * gradW_ij;
}

__device__ __forceinline__ double getUStar_Riemann(double U_L, double P_L, double rho_L, double U_R, double P_R, double rho_R, double c)
{
	return (rho_L * U_L + rho_R * U_R + (P_L - P_R) / c) / (rho_L + rho_R);
}

__device__ __forceinline__ double3 getAverageVelocity(double rho_i, double rho_j, double3 v_i, double3 v_j)
{
	return (rho_i * v_i + rho_j * v_j) / (rho_i + rho_j);
}

__device__ __forceinline__ double3 getVelocityStar_Riemann(double U_L, double rho_L, double U_R, double rho_R, double3 v_bar, double U_star, double3 e_ij)
{
	e_ij *= -1.0;
	return (U_star - (rho_L * U_L + rho_R * U_R) / (rho_L + rho_R)) * e_ij + v_bar;
}

__device__ __forceinline__ double getDensityIncrementFromParticleJ_Riemann(double rho_i, double3 v_i, double3 v_star, double V_j, double3 gradW_ij)
{
	return 2. * rho_i * V_j * dot(v_i - v_star, gradW_ij);
}

__device__ __forceinline__ double getPStar_Riemann(double U_L, double P_L, double rho_L, double U_R, double P_R, double rho_R, double beta)
{
	return (rho_L * P_R + rho_R * P_L + rho_L * rho_R * beta * (U_L - U_R)) / (rho_L + rho_R);
}

__device__ __forceinline__ double3 getPressureAccelerationFromParticleJ_Riemann(double m_i, double V_i, double V_j, double P_star, double3 gradW_ij)
{
	return -2. / m_i * V_i * V_j * P_star * gradW_ij;
}

__device__ __forceinline__ double3 getViscosityAccelerationFromParticleJ_Riemann(double m_i, double V_i, double3 v_i, double V_j, double3 v_j, double gamma_ij, double absR_ij, double3 e_ij, double3 gradW_ij)
{
	return 2. / m_i * V_i * V_j * gamma_ij * (v_i - v_j) / absR_ij * dot(e_ij, gradW_ij);
}

__device__ __forceinline__ int ParallelBondedContact(double& bondNormalForce, double& bondTorsionalTorque, double3& bondShearForce, double3& bondBendingTorque,
	double3 contactNormalPrev,
	double3 contactNormal,
	double3 relativeVelocityAtContact,
	double3 angularVelocityA,
	double3 angularVelocityB,
	double radiusA,
	double radiusB,
	double timeStep,
	double bondMultiplier,
	double bondElasticModulus,
	double bondStiffnessRatioNormalToShear,
	double bondTensileStrength,
	double bondCohesion,
	double bondFrictionCoefficient)
{
	double3 theta1 = cross(contactNormalPrev, contactNormal);
	bondShearForce = rotateVector(bondShearForce, theta1);
	bondBendingTorque = rotateVector(bondBendingTorque, theta1);
	double3 theta2 = dot(0.5 * (angularVelocityA + angularVelocityB) * timeStep, contactNormal) * contactNormal;
	bondShearForce = rotateVector(bondShearForce, theta2);
	bondBendingTorque = rotateVector(bondBendingTorque, theta2);
	double minRadius = radiusA < radiusB ? radiusA : radiusB;
	double bondRadius = bondMultiplier * minRadius;
	double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius / 4. * pi();// inertia moment
	double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment
	double normalStiffnessUnitArea = bondElasticModulus / (radiusA + radiusB);
	double shearStiffnessUnitArea = normalStiffnessUnitArea / bondStiffnessRatioNormalToShear;

	double3 normalIncrement = dot(relativeVelocityAtContact, contactNormal) * contactNormal * timeStep;
	double3 tangentialIncrement = relativeVelocityAtContact * timeStep - normalIncrement;
	bondNormalForce -= dot(normalIncrement * normalStiffnessUnitArea * bondArea, contactNormal);
	bondShearForce -= tangentialIncrement * shearStiffnessUnitArea * bondArea;
	double3 relativeAngularVelocity = angularVelocityA - angularVelocityB;
	normalIncrement = dot(relativeAngularVelocity, contactNormal) * contactNormal * timeStep;
	tangentialIncrement = relativeAngularVelocity * timeStep - normalIncrement;
	bondTorsionalTorque -= dot(normalIncrement * shearStiffnessUnitArea * bondPolarInertiaMoment, contactNormal);
	bondBendingTorque -= tangentialIncrement * normalStiffnessUnitArea * bondInertiaMoment;

	double maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	double maxShearStress = length(bondShearForce) / bondArea + abs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress

	int isBonded = 1;
	if (bondTensileStrength > 0 && maxNormalStress > bondTensileStrength)
	{
		isBonded = 0;
	}
	else if (bondCohesion > 0 && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		isBonded = 0;
	}
	return isBonded;
}

static __device__ __forceinline__ double3 integrateSlidingOrRollingSpring(const double3 springPrev, double3 springVelocity, double3 contactNormal, double3 normalContactForce, double frictionCoefficient, double stiffness, double dampingCoefficient, double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0)
	{
		double3 springPrev1 = springPrev - dot(springPrev, contactNormal) * contactNormal;
		double absoluteSpringPrev1 = length(springPrev1);
		if (absoluteSpringPrev1 > 1.e-10)
		{
			springPrev1 *= length(springPrev) / absoluteSpringPrev1;
		}
		spring = springPrev1 + springVelocity * timeStep;
		double3 springForce = -stiffness * spring - dampingCoefficient * springVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * springVelocity) / stiffness;
		}
	}
	return spring;
}

static __device__ __forceinline__ double3 integrateTorsionSpring(const double3 springPrev, double3 torsionRelativeVelocity, double3 contactNormal, double3 normalContactForce, double frictionCoefficient, double stiffness, double dampingCoefficient, double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0)
	{
		spring = dot(springPrev + torsionRelativeVelocity * timeStep, contactNormal) * contactNormal;
		double3 springForce = -stiffness * spring - dampingCoefficient * torsionRelativeVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * torsionRelativeVelocity) / stiffness;
		}
	}
	return spring;
}

__device__ __forceinline__ void LinearContact(double3& contactForce, double3& contactTorque, double3& slidingSpring, double3& rollingSpring, double3& torsionSpring,
	double3 relativeVelocityAtContact,
	double3 relativeAngularVelocityAtContact,
	double3 contactNormal,
	double normalOverlap,
	double effectiveMass,
	double effectiveRadius,
	double timeStep,
	double normalStiffness,
	double slidingStiffness,
	double rollingStiffness,
	double torsionStiffness,
	double normalDissipation,
	double slidingDissipation,
	double rollingDissipation,
	double torsionDissipation,
	double slidingFrictionCoefficient,
	double rollingFrictionCoefficient,
	double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		double normalDampingCoefficient = 2. * normalDissipation * sqrt(effectiveMass * normalStiffness);
		double slidingDampingCoefficient = 2. * slidingDissipation * sqrt(effectiveMass * slidingStiffness);
		double rollingDampingCoefficient = 2. * rollingDissipation * sqrt(effectiveMass * rollingStiffness);
		double torsionDampingCoefficient = 2. * torsionDissipation * sqrt(effectiveMass * torsionStiffness);

		double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, slidingRelativeVelocity, contactNormal, normalContactForce, slidingFrictionCoefficient, slidingStiffness, slidingDampingCoefficient, timeStep);
		double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, rollingRelativeVelocity, contactNormal, normalContactForce, rollingFrictionCoefficient, rollingStiffness, rollingDampingCoefficient, timeStep);
		double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		double effectiveDiameter = 2 * effectiveRadius;
		double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, torsionRelativeVelocity, contactNormal, normalContactForce, torsionFrictionCoefficient, torsionStiffness, torsionDampingCoefficient, timeStep);
		double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}

__device__ __forceinline__ void HertzianMindlinContact(double3& contactForce, double3& contactTorque, double3& slidingSpring, double3& rollingSpring, double3& torsionSpring,
	double3 relativeVelocityAtContact,
	double3 relativeAngularVelocityAtContact,
	double3 contactNormal,
	double normalOverlap,
	double effectiveMass,
	double effectiveRadius,
	double timeStep,
	double effectiveElasticModulus,
	double effectiveShearModulus,
	double dissipation,
	double stiffnessRatioRollingToSliding,
	double stiffnessRatioTorsionToSliding,
	double slidingFrictionCoefficient,
	double rollingFrictionCoefficient,
	double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		double normalStiffness = 4. / 3. * effectiveElasticModulus * sqrt(effectiveRadius * normalOverlap);
		double slidingStiffness = 8. * effectiveShearModulus * sqrt(effectiveRadius * normalOverlap);
		double normalDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * normalStiffness);
		double slidingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * slidingStiffness);

		double rollingStiffness = slidingStiffness * stiffnessRatioRollingToSliding;
		double torsionStiffness = slidingStiffness * stiffnessRatioTorsionToSliding;
		double rollingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * rollingStiffness);
		double torsionDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * torsionStiffness);

		double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, slidingRelativeVelocity, contactNormal, normalContactForce, slidingFrictionCoefficient, slidingStiffness, slidingDampingCoefficient, timeStep);
		double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, rollingRelativeVelocity, contactNormal, normalContactForce, rollingFrictionCoefficient, rollingStiffness, rollingDampingCoefficient, timeStep);
		double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		double effectiveDiameter = 2 * effectiveRadius;
		double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, torsionRelativeVelocity, contactNormal, normalContactForce, torsionFrictionCoefficient, torsionStiffness, torsionDampingCoefficient, timeStep);
		double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}