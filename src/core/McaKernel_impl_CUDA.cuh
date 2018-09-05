/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
//#include "math_functions.h"
#include "McaParams.h"
#include "../tools/helper_double3.h"

__constant__ McaParams params;

__device__
void print_double3(const char* txt, double3 vec) {
	printf("%s: %3.6f, %3.6f, %3.6f\n", txt, vec.x, vec.y, vec.z);
}

struct integrate_verlet_functor {

	__host__ __device__
	integrate_verlet_functor() {
	}

	template<typename Tuple>
	__device__
	void operator()(Tuple t) {
		volatile double4 posData = thrust::get<0>(t);
		volatile double4 posPrevData = thrust::get<1>(t);
		volatile double4 velData = thrust::get<2>(t);
		volatile double4 accelData = thrust::get<3>(t);

		volatile double4 thetaData = thrust::get<4>(t);
		volatile double4 thetaPrevData = thrust::get<5>(t);
		volatile double4 thetaAccelData = thrust::get<6>(t);

		double3 pos = make_double3(posData);
		double3 pos_prev = make_double3(posPrevData);
		double3 vel = make_double3(velData);
		double3 accel = make_double3(accelData);

		double3 theta = make_double3(thetaData);
		double3 thetaPrev = make_double3(thetaPrevData);
		double3 thetaAccel = make_double3(thetaAccelData);

		// set this to zero to disable collisions with cube sides
#if 0
		if (pos.x > params.spaceSize.x - params.particleRadius) {
			accel.x = -2.0 * vel.x / deltaTime * params.boundaryDamping;
		}

		if (pos.x < params.worldOrigin.x + params.particleRadius) {
			accel.x = -2.0 * vel.x / deltaTime * params.boundaryDamping;
		}

		if (pos.y > params.spaceSize.y - params.particleRadius) {
			accel.y = -2.0 * vel.y / deltaTime * params.boundaryDamping;
		}
		if (pos.y < params.worldOrigin.y + params.particleRadius) {
			accel.y = -2.0 * vel.y / deltaTime * params.boundaryDamping;
		}

#endif

//todo

		/* double _2pi = 2 * CUDART_PI;
		 if (theta.z > _2pi)
		 theta.z -= _2pi * ((int) (theta.z/_2pi));
		 if (theta.z < -_2pi)
		 theta.z -= _2pi * ((int) (theta.z/_2pi));
		if (theta.z > _2pi)
		 theta.z -= _2pi;
		 if (theta.z < -_2pi)
		 theta.z += _2pi;*/


		double3 tempPos = pos;
		double3 tempTheta = theta;

		// new position = 2 * old position -  old_prev_position + accel * deltaTime * deltaTime
		//printf("pos.x = %f, pos.y = %f, pos_pref.x = %f, pos_prev.y = %f \n", pos.x, pos.y, pos_prev.x, pos_prev.y);
		pos = 2.0 * pos - pos_prev + accel * params.timeStep * params.timeStep;

		theta = 2.0 * theta - thetaPrev
				+ thetaAccel * params.timeStep * params.timeStep;

		//printf("theta.x = %3.4f theta.y = %3.4f\n", theta.x, theta.y);

		//vel_prev = (new_position - pre_old_position) / (2 * deltaTime)
		vel = 1.0 / (2.0 * params.timeStep) * (pos - pos_prev);

		pos_prev = tempPos;
		thetaPrev = tempTheta;

		// store new position and velocity
		thrust::get<0>(t) = make_double4(pos, posData.w);
		thrust::get<1>(t) = make_double4(pos_prev, posPrevData.w);
		thrust::get<2>(t) = make_double4(vel, velData.w);
		thrust::get<3>(t) = make_double4(accel, accelData.w);

		thrust::get<4>(t) = make_double4(theta, thetaData.w);
		thrust::get<5>(t) = make_double4(thetaPrev, thetaPrevData.w);
		thrust::get<6>(t) = make_double4(thetaAccel, thetaAccelData.w);
	}
};

__global__
void integrateVerletFunctionD(double4 *posData, double4 *posPrevData, double4 *velData, double4 *accelData,
		double4 *thetaData,
		double4 *thetaPrevData,
		double4 *thetaAccelData, uint numParticles) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	double3 pos = make_double3(posData[index]);
	double3 pos_prev = make_double3(posPrevData[index]);
	double3 vel = make_double3(velData[index]);
	double3 accel = make_double3(accelData[index]);

	double3 theta = make_double3(thetaData[index]);
	double3 thetaPrev = make_double3(thetaPrevData[index]);
	double3 thetaAccel = make_double3(thetaAccelData[index]);

	double3 tempPos = pos;
	double3 tempTheta = theta;

	// new position = 2 * old position -  old_prev_position + accel * deltaTime * deltaTime
	//printf("pos.x = %f, pos.y = %f, pos_pref.x = %f, pos_prev.y = %f \n", pos.x, pos.y, pos_prev.x, pos_prev.y);
	pos = 2.0 * pos - pos_prev + accel * params.timeStep * params.timeStep;

	theta = 2.0 * theta - thetaPrev
			+ thetaAccel * params.timeStep * params.timeStep;

	//printf("theta.x = %3.4f theta.y = %3.4f\n", theta.x, theta.y);

	//vel_prev = (new_position - pre_old_position) / (2 * deltaTime)
	vel = 1.0 / (2.0 * params.timeStep) * (pos - pos_prev);

	pos_prev = tempPos;
	thetaPrev = tempTheta;

	posData[index] = make_double4(pos, posData[index].w);
	posPrevData[index] = make_double4(pos_prev, 0.0);
	velData[index] = make_double4(vel, 0.0);
	accelData[index] = make_double4(accel, 0.0);
	thetaData[index] = make_double4(theta, 0.0);
	thetaPrevData[index] = make_double4(thetaPrev, 0.0);
	thetaAccelData[index] = make_double4(thetaAccel, 0.0);
}

// calculate position in uniform grid
__device__ int3 calcGridPos(double3 p) {
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos) {
	return (gridPos.x * 73856093 + gridPos.y * 19349663 + gridPos.z * 83492791)
			% params.numGridCells;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint *gridParticleHash,  // output
		uint *gridParticleIndex, // output
		double4 *pos,               // input: positions
		uint numParticles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	volatile double4 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// find the start of each cell in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint *cellStart,   // output: cell start index
		uint *cellEnd,          // output: cell end index
		uint *gridParticleHash, // input: sorted grid hashes
		uint *gridParticleIndex, // input: sorted particle indices
		uint numParticles) {
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1) {
			cellEnd[hash] = index + 1;
		}
	}
}


__device__
inline bool isLinked(uint* neighbours, uint* numNeighb, uint index, uint neighbIndex, int& indexInNeighbrs) {

	//printf("isLinked:: index = %d, numNeighb = %d, neighbIndex = %d\n", index, numNeighb[index], neighbIndex);
#pragma unroll
	for (int i = 0; i < params.mcaMaxNumNeighbors; i++) {
		if (i == numNeighb[index])
			return false;
		if (neighbours[index * params.mcaMaxNumNeighbors + i] == neighbIndex) {
			indexInNeighbrs = i;
			return true;
		}
	}
	return false;
}

__device__
inline void checkLink(uint* neighbours, uint* numNeighb, uint index, uint neighbIndex,
		double3 posA, double3 posB, double *deformation) {
	double dist = length(posB - posA);
	double delta = dist - params.particleRadius * 2.0;

	int indexInNeighbrs = -1;
	if (isLinked(neighbours, numNeighb, index, neighbIndex, indexInNeighbrs)) {
		if (delta > params.mcaRmax) {

			if (numNeighb[index] == 0) {
				printf("remove link ERROR!!! numNeighb[%d] < 0\n", index);
			}

			//remove link
			neighbours[index * params.mcaMaxNumNeighbors + indexInNeighbrs] =
					neighbours[index * params.mcaMaxNumNeighbors + --numNeighb[index]];

			return;
		}

		deformation[index] += (dist - params.automateD)
						/ params.automateD;
		return;
	} else if (params.mcaIsBrittle) {
		return;
	} else {
		if (delta <= params.mcaRmin) {
			if (numNeighb[index] == params.mcaMaxNumNeighbors) {
				printf("add link ERROR!!! numNeighb[%d] == MAX_NUM_NEIGHBORS\n",
						index);
				return;
			}
			// add link
			neighbours[index * params.mcaMaxNumNeighbors + numNeighb[index]++] =
					neighbIndex;

			deformation[index] += (dist - params.automateD)
									/ params.automateD;
			return;
		}
	}
}

__device__
inline double3 getNeighbrsNormForce(uint index, uint neighbIndex, double4* positions,
		double *deformations, uint* neighbrs, uint* numNeighbrs, double3 norm) {
	double3 force = make_double3(0);
	double3 posB = make_double3(positions[index]);

	for (int i = 0; i < numNeighbrs[index]; i++) {
		if (neighbIndex == neighbrs[index * params.mcaMaxNumNeighbors + i])
			continue;
		double3 posA = make_double3(
				positions[neighbrs[index * params.mcaMaxNumNeighbors + i]]);
		double3 relPos = posA - posB;
		double dist = length(relPos);
		double3 norm2 = relPos / dist;
		double S = params.automateS;
		/*double S;
		double q = dist / 2.0;
		if (q > params.particleRadius) {
			S = 0;
		} else {
			//S = 4 * (params.particleRadius * params.particleRadius - q * q);
			S = params.automateD * 2
					* sqrt(
							params.particleRadius * params.particleRadius
									- q * q);
		}*/
		double deformation = (dist - params.automateD)
				/ params.automateD;
		force += (2.0 * params.mcaG * (deformation - deformations[index])) * S
				* dot(norm, norm2) * norm;
	}

	posB = make_double3(positions[neighbIndex]);
	for (int i = 0; i < numNeighbrs[neighbIndex]; i++) {
		if (index == neighbrs[neighbIndex * params.mcaMaxNumNeighbors + i])
			continue;
		double3 posA =
				make_double3(
						positions[neighbrs[neighbIndex
								* params.mcaMaxNumNeighbors + i]]);
		double3 relPos = posA - posB;
		//double3 relPos =  posA - posB;
		double dist = length(relPos);

		double3 norm2 = relPos / dist;
		double S = params.automateS;
		/*double S;
		double q = dist / 2.0;
		if (q > params.particleRadius) {
			S = 0;
		} else {
			//S = 4 * (params.particleRadius * params.particleRadius - q * q);
			S = params.automateD * 2
					* sqrt(
							params.particleRadius * params.particleRadius
									- q * q);
		}*/
		double deformation = (dist - params.automateD)
				/ params.automateD;

		force += (2.0 * params.mcaG * (deformation - deformations[neighbIndex]))
				* S * dot(norm, norm2) * norm;
	}

	return force;
}
/*
__device__
inline double3 mcaSpheres(uint index, uint neighbIndex, double4 *positions,
		double4 *prevPositions, double4 *velocities, double4 *angles,
		double4 *prevAngles, double4 *angAccelerations, double *deformations,
		double radiusA, double radiusB, uint* neighbrs, uint* numNeighb) {

	double3 posA = make_double3(positions[index]);
	double3 posB = make_double3(positions[neighbIndex]);
	double3 prevPosA = make_double3(prevPositions[index]);
	double3 prevPosB = make_double3(prevPositions[neighbIndex]);
	double3 velA = make_double3(velocities[index]);
	double3 velB = make_double3(velocities[neighbIndex]);
	double3 angleA = make_double3(angles[index]);
	double3 angleB = make_double3(angles[neighbIndex]);
	double3 prevAngleA = make_double3(prevAngles[index]);
	double3 prevAngleB = make_double3(prevAngles[neighbIndex]);

	// calculate relative position
	double3 relPos = posB - posA;

	double dist = length(relPos);
	//double collideDist = radiusA + radiusB;

	double S = params.automateS;

	double q = dist / 2.0;

	double S;
	if (q > params.particleRadius) {
		S = 0;
	} else {
		//S = 4 * (params.particleRadius * params.particleRadius - q * q);
		S = params.automateD * 2 * sqrt(params.particleRadius * params.particleRadius - q * q);
	}

	double3 accel = make_double3(0);
	double deformation = (dist - params.automateD)
			/ params.automateD;

	//if (dist < collideDist)
	//{
		double3 norm = relPos / dist;

		double3 prevRelPos = prevPosB - prevPosA;
		double3 prevNorm = prevRelPos / length(prevRelPos);

		// relative velocity
		double3 relVel = velB - velA;

		// relative tangential velocity
		double3 tanVel = relVel - (dot(relVel, norm) * norm);

		//NORMAL ACCELERATIONS
		accel += (2.0 * params.mcaG * (deformation - deformations[index])) * S
				* norm
				+ getNeighbrsNormForce(index, neighbIndex, positions,
						deformations, neighbrs, numNeighb, norm);
		accel += -(params.mcaK * params.mcaD * -deformations[index]) * S * norm;

		//TANGETIAL ACCELERATIONS
		// relative angle of pair
		double3 relThetaOfPair = acos(dot(norm, prevNorm))
				/ sqrt(1.0 - dot(norm, prevNorm) * dot(norm, prevNorm))
				* cross(norm, prevNorm);
		double3 relThetaA = angleA - prevAngleA;
		double3 relThetaB = angleB - prevAngleB;

		if (isnan(relThetaOfPair.x))
			accel += -1.0
					* (params.mcaG * (-1.0 * cross(relThetaA, norm))
							+ params.mcaG * (-1.0 * cross(relThetaB, norm)));
		else
			accel +=
					-1.0
							* (params.mcaG
									* (cross(relThetaOfPair - relThetaA, norm))
									+ params.mcaG
											* (cross(relThetaOfPair - relThetaB,
													norm)));

		// relative tangential accel
		double3 tanAccel = accel - (dot(accel, norm) * norm);
		angAccelerations[index] += make_double4(cross(tanAccel, norm) / q, 0);

		if (length(accel) > 100000000) {
			printf("|tanAccel| = %f\n",length(tanAccel));
			printf("|accel| = %f\n",length(accel));
			printf("|normAccel| = %f\n",length(accel - tanAccel));
			printf("deformation = %f\n", deformation);
			printf("totaldeformation = %f\n", deformations[index]);
		}

		if (index == 0 && neighbIndex == 1) {
		 print_double3("relThetaOfPair", relThetaOfPair);
		 print_double3("angleA", angleA);
		 print_double3("angleB", angleB);
		 print_double3("accel", accel);
		 print_double3("tanAccel", tanAccel);
		 print_double3("angAccel", make_double3(angAccelerations[index]));
		 }

		uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		 printf("part[%d], iteration = %d, dist = %3.6f collideDist = %3.6f\n", index, iteration,
		 dist, collideDist);
	//}

	static uint iterator = 0;
	 iterator++;
	 if (iterator <= 50000) {
	 printf("\n");
	 printf("index = %d\n", index);
	 printf("neighb_index = %d\n", neighbIndex);
	 print_double3("posA:", posA);
	 print_double3("posB:", posA);
	 print_double3("posPrevA:", prevPosA);
	 print_double3("posPrevB:", prevPosB);
	 print_double3("norm:", norm);
	 print_double3("prevNorm:", prevNorm);
	 print_double3("angleA:", angleA);
	 print_double3("angleB:", angleB);
	 print_double3("relTheta:", relTheta);
	 print_double3("angAccelerationA:",
	 make_double3(angAccelerations[index]));
	 print_double3("angAccelerationB:",
	 make_double3(angAccelerations[neighbIndex]));
	 print_double3("tanAccel:", tanAccel);
	 print_double3("accel:", accel);
	 }

	return accel;
}*/

__device__
inline double3 mcaSpheresStress(uint index, uint neighbIndex, double4 *positions,
		double4 *prevPositions, double4 *velocities, double4 *angles,
		double4 *prevAngles, double4 *angAccelerations, double *deformations,
		double radiusA, double radiusB, uint* neighbrs, uint* numNeighb, double* stress) {

	double3 posA = make_double3(positions[index]);
	double3 posB = make_double3(positions[neighbIndex]);
	double3 prevPosA = make_double3(prevPositions[index]);
	double3 prevPosB = make_double3(prevPositions[neighbIndex]);
	double3 velA = make_double3(velocities[index]);
	double3 velB = make_double3(velocities[neighbIndex]);
	double3 angleA = make_double3(angles[index]);
	double3 angleB = make_double3(angles[neighbIndex]);
	double3 prevAngleA = make_double3(prevAngles[index]);
	double3 prevAngleB = make_double3(prevAngles[neighbIndex]);

	// calculate relative position
	double3 relPos = posB - posA;

	double dist = length(relPos);
	//double collideDist = radiusA + radiusB;

	double S = params.automateS;

	double q = dist / 2.0;

	double3 accel = make_double3(0);
	double deformation = (dist - params.automateD)
			/ params.automateD;

		double3 norm = relPos / dist;

		double3 prevRelPos = prevPosB - prevPosA;
		double3 prevNorm = prevRelPos / length(prevRelPos);

		// relative velocity
		double3 relVel = velB - velA;

		// relative tangential velocity
		double3 tanVel = relVel - (dot(relVel, norm) * norm);

		//NORMAL ACCELERATIONS
		accel += (2.0 * params.mcaG * (deformation - deformations[index])) * S
				* norm
				+ getNeighbrsNormForce(index, neighbIndex, positions,
						deformations, neighbrs, numNeighb, norm);
		accel += -(params.mcaK * params.mcaD * -deformations[neighbIndex]) * S * norm;

		stress[index] += length(accel) / S;

		//TANGETIAL ACCELERATIONS
		// relative angle of pair
		double3 relThetaOfPair = acos(dot(norm, prevNorm))
				/ sqrt(1.0 - dot(norm, prevNorm) * dot(norm, prevNorm))
				* cross(norm, prevNorm);
		double3 relThetaA = angleA - prevAngleA;
		double3 relThetaB = angleB - prevAngleB;

		if (isnan(relThetaOfPair.x))
			accel += -1.0
					* (params.mcaG * (-1.0 * cross(relThetaA, norm))
							+ params.mcaG * (-1.0 * cross(relThetaB, norm)));
		else
			accel +=
					-1.0 * (params.mcaG
									* (cross(relThetaOfPair - relThetaA, norm))
									+ params.mcaG
											* (cross(relThetaOfPair - relThetaB,
													norm)));

		// tangential accel
		double3 tanAccel = accel - (dot(accel, norm) * norm);
		angAccelerations[index] += make_double4(cross(tanAccel, norm) / q, 0);

	return accel;
}

__device__
inline double3 collideSpheres(uint index, uint neighbIndex, double4 *positions,
		double4 *velocities, double *deformations, double radiusA,
		double radiusB) {
	double3 posA = make_double3(positions[index]);
	double3 posB = make_double3(positions[neighbIndex]);
	double3 velA = make_double3(velocities[index]);
	double3 velB = make_double3(velocities[neighbIndex]);

	// calculate relative position
	double3 relPos = posB - posA;

	double dist = length(relPos);
	//double collideDist = radiusA + radiusB + params.mcaDelta;
	double collideDist = radiusA + radiusB;

	double3 accel = make_double3(0.0);

	if (dist < collideDist) {
		double3 norm = relPos / dist;

		double3 relVel = velB - velA;
		// relative tangential accel
		double3 normVel = dot(relVel, norm) * norm;
		double3 tanVel = relVel - normVel;
		double3 tanNorm = tanVel / length(tanVel);

		double q = dist / 2.0;
		double S = params.automateD * 2 * sqrt(params.particleRadius * params.particleRadius - q * q);

		double absN = S * length(normVel) / params.timeStep;

		// tangential shear force
		accel += -params.mcaShear * absN * tanNorm;

		// spring force

		accel += norm * 0.5f
						* ((1.0 + params.bounceCollision) * dot(relVel, norm)
								- (collideDist - dist) / params.timeStep) / params.timeStep;
	}

	return accel;
}

__device__
double3 collideCollider(double3 posA, double3 posB, double3 velA, double3 velB,
		double radiusA, double radiusB, double attraction) {

	// calculate relative position
	double3 relPos = posB - posA;

	double dist = length(relPos);
	double collideDist = radiusA + radiusB;

	double3 accel = make_double3(0.0);

	if (dist < collideDist) {
		double3 norm = relPos / dist;

		// relative velocity
		double3 relVel = velB - velA;

		// relative tangential accel
		double3 tanVel = relVel - (dot(relVel, norm) * norm);

		// spring force
		accel = -params.spring * (collideDist - dist) * norm;
		// tangential shear force
		accel += params.mcaShear * tanVel;
		// attraction
		accel += attraction * relPos;

		/*uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

		 printf("part[%d], iteration = %d, dist = %3.6f collideDist = %3.6f\n", index, iteration,
		 dist, collideDist);*/
	}

	return accel;
}

/*// collide two spheres using DEM method
 __device__
 double3 collideSpheres(double3 posA, double3 posB, double3 velA, double3 velB,
 double radiusA, double radiusB, double attraction) {
 static int iteration = 0;
 iteration++;

 // calculate relative position
 double3 relPos = posB - posA;

 double dist = length(relPos);
 double collideDist = radiusA + radiusB;

 double3 force = make_double3(0.0);

 if (dist < collideDist) {
 double3 norm = relPos / dist;

 // relative velocity
 double3 relVel = velB - velA;

 // relative tangential velocity
 double3 tanVel = relVel - (dot(relVel, norm) * norm);

 // spring force
 force = -params.spring * (collideDist - dist) * norm;
 // dashpot (damping) force
 force += params.damping * relVel;
 // tangential shear force
 force += params.shear * tanVel;
 // attraction
 force += attraction * relPos;

 uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

 printf("part[%d], iteration = %d, dist = %3.6f collideDist = %3.6f\n", index, iteration,
 dist, collideDist);
 }

 return force;
 }*/

// collide a particle against all other particles in a given cell
/*__device__
inline double3 collideCell(uint* gridParticlesIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double4 *oldPositions, double4 *prevPositions,
		double4 *velocity, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, uint *cellStart, uint *cellEnd,
		uint* neighbrs, uint* numNeighb, double* deformation) {
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	double3 force = make_double3(0.0);

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex)          // check not colliding with self
					{
				uint originalIndexNeighb = gridParticlesIndex[j];

				int indexInNeighb;
				bool bIsLinked = isLinked(neighbrs, numNeighb, originalIndex,
						originalIndexNeighb, indexInNeighb);
				if (bIsLinked) {
					//printf("linked\n");
					//continue;
					force += mcaSpheres(originalIndex, originalIndexNeighb,
							oldPositions, prevPositions, velocity, angles,
							prevAngles, angAccelerations, deformation,
							params.particleRadius, params.particleRadius,
							neighbrs, numNeighb);

				} else {
					// collide two spheres
					force += collideSpheres(originalIndex, originalIndexNeighb,
							oldPositions, velocity, deformation,
							params.particleRadius, params.particleRadius);
				}
			}
		}
	}

	return force;
}*/

__device__
inline double3 collideCellStress(uint* gridParticlesIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double4 *oldPositions, double4 *prevPositions,
		double4 *velocity, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, uint *cellStart, uint *cellEnd,
		uint* neighbrs, uint* numNeighb, double* deformation, double* stress) {
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	double3 force = make_double3(0.0);

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex)             // check not colliding with self
					{
				uint originalIndexNeighb = gridParticlesIndex[j];

				int indexInNeighb;
				bool bIsLinked = isLinked(neighbrs, numNeighb, originalIndex,
						originalIndexNeighb, indexInNeighb);
				if (bIsLinked) {
					//printf("linked\n");
					//continue;
					force += mcaSpheresStress(originalIndex,
							originalIndexNeighb, oldPositions, prevPositions,
							velocity, angles, prevAngles, angAccelerations,
							deformation, params.particleRadius,
							params.particleRadius, neighbrs, numNeighb, stress);

				} else {
					// collide two spheres
					force += collideSpheres(originalIndex, originalIndexNeighb,
							oldPositions, velocity, deformation,
							params.particleRadius, params.particleRadius);
				}
			}
		}
	}

	return force;
}

__device__
inline void updateLink(uint *gridParticleIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double3 oldPos, double4 *oldPosition,
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb,
		double *deformation) {

	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex)              // check not checking with self
					{
				uint originalIndexNeighb = gridParticleIndex[j];
				double3 oldPos2 = make_double3(
						oldPosition[originalIndexNeighb]);

				checkLink(neighbrs, numNeighb, originalIndex,
						originalIndexNeighb, oldPos, oldPos2, deformation);
			}
		}
	}
}

__device__
inline void addInitLink(uint* neighbours, uint* numNeighb, uint index, uint neighbIndex,
		double3 posA, double3 posB) {
	double dist = length(posB - posA);
	double delta = dist - params.particleRadius * 2.0;

	if ((delta < params.mcaRmax) && (delta > params.mcaRmin)) {
		if (numNeighb[index] == params.mcaMaxNumNeighbors) {
			printf("ERROR!!! numNeighb[%d] == MAX_NUM_NEIGHBORS\n", index);
			return;
		}

		neighbours[index * params.mcaMaxNumNeighbors + numNeighb[index]++] =
				neighbIndex;
	}
}

__device__
inline void addInitLink2(uint* neighbours, uint* numNeighb, uint index, uint neighbIndex,
		double3 posA, double3 posB) {
	double dist = length(posB - posA);
	double delta = dist - params.particleRadius * 2.0;

	if ((delta < params.mcaRmax) && (delta > params.mcaRmin)) {
		if (numNeighb[index] == params.mcaMaxNumNeighbors) {
			printf("ERROR!!! numNeighb[%d] == MAX_NUM_NEIGHBORS\n", index);
			return;
		}

		neighbours[index * params.mcaMaxNumNeighbors + atomicAdd(&numNeighb[index],1)] =
				neighbIndex;
	}
}

__device__
inline void initLink(uint *gridParticleIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double3 oldPos, double4 *oldPosition,
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb) {

	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex) // check not colliding with self
					{
				uint originalIndexNeighb = gridParticleIndex[j];
				double3 oldPos2 = make_double3(
						oldPosition[originalIndexNeighb]);

				addInitLink(neighbrs, numNeighb, originalIndex,
						originalIndexNeighb, oldPos, oldPos2);
			}
		}
	}
}

__device__
inline void initLink2(uint *gridParticleIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double3 oldPos, double4 *oldPosition,
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb) {

	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex) // check not colliding with self
			{
				uint originalIndexNeighb = gridParticleIndex[j];
				double3 oldPos2 = make_double3(oldPosition[originalIndexNeighb]);

				addInitLink2(neighbrs, numNeighb, originalIndex,
										originalIndexNeighb, oldPos, oldPos2);
			}
		}
	}
}


__global__
void updateLinksAndDeformD(uint *gridParticleIndex, double * deformation,
		double4 *oldPositions, double4 *colorLinks, double4 *colorDeform,
		uint *cellStart, uint *cellEnd, uint* neighbours, uint* numNeighb,
		uint numParticles) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;
	uint originalIndex = gridParticleIndex[index];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighbourCellPos = gridPos + make_int3(x, y, z);
				updateLink(gridParticleIndex, neighbourCellPos, index,
						originalIndex, oldPos, oldPositions, cellStart, cellEnd,
						neighbours, numNeighb, deformation);
			}
		}

	deformation[originalIndex] = deformation[originalIndex] / params.mcaMaxNumNeighbors;
	/*if (numNeighb[originalIndex]) {
		deformation[originalIndex] = deformation[originalIndex]
				/ numNeighb[originalIndex];
	}*/

	if (colorDeform) {
		double k1 = 1.0 - abs(deformation[originalIndex] * 50.0);
		colorDeform[originalIndex].x = 1.0;
		colorDeform[originalIndex].y = k1;
		colorDeform[originalIndex].z = k1;
		colorDeform[originalIndex].w = 0.5;

		double k2 = (double) numNeighb[originalIndex]
				/ (double) params.mcaMaxNumNeighbors;
		colorLinks[originalIndex].x = k2;
		colorLinks[originalIndex].y = k2;
		colorLinks[originalIndex].z = k2;
		colorLinks[originalIndex].w = 0.5;
	}

}


__global__
void updateLinksAndDeformChangedD(uint *gridParticleIndex, double * deformation,
		double4 *oldPositions, double4 *colorLinks, double4 *colorDeform,
		uint *cellStart, uint *cellEnd, uint* neighbours, uint* numNeighb,
		uint numParticles) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles * 27)
		return;

	int i = (int) index / 27;

	uint originalIndex = gridParticleIndex[i];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	int z = (index % 3) - 1;
	int y = ((index / 3) % 3) - 1;
	int x = ((index / (3 * 3)) % 3) - 1;

	int3 neighbourCellPos = gridPos + make_int3(x, y, z);
	updateLink(gridParticleIndex, neighbourCellPos, i, originalIndex,
			oldPos, oldPositions, cellStart, cellEnd, neighbours, numNeighb,
			deformation);

	deformation[originalIndex] = deformation[originalIndex] / params.mcaMaxNumNeighbors;
	/*if (numNeighb[originalIndex]) {
		deformation[originalIndex] = deformation[originalIndex]
				/ numNeighb[originalIndex];
	}*/

	if (colorDeform) {
		double k1 = 1.0 - abs(deformation[originalIndex] * 50.0);
		colorDeform[originalIndex].x = 1.0;
		colorDeform[originalIndex].y = k1;
		colorDeform[originalIndex].z = k1;
		colorDeform[originalIndex].w = 0.5;

		double k2 = (double) numNeighb[originalIndex]
				/ (double) params.mcaMaxNumNeighbors;
		colorLinks[originalIndex].x = k2;
		colorLinks[originalIndex].y = k2;
		colorLinks[originalIndex].z = k2;
		colorLinks[originalIndex].w = 0.5;
	}

}


__global__
void initLinksD(uint *gridParticleIndex,
		double4 *oldPositions, uint *cellStart, uint *cellEnd,
		uint* neighbours, uint* numNeighb,
		uint numParticles) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;
	uint originalIndex = gridParticleIndex[index];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighbourCellPos = gridPos + make_int3(x, y, z);
				initLink(gridParticleIndex, neighbourCellPos, index,
						originalIndex, oldPos, oldPositions, cellStart, cellEnd,
						neighbours, numNeighb);
			}
		}
}

__global__
void initLinksChangedD(uint *gridParticleIndex,
		double4 *oldPositions, uint *cellStart, uint *cellEnd,
		uint* neighbours, uint* numNeighb,
		uint numParticles) {

	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//qw
	/*if (threadIdx.x > 26) return;
	int index = blockIdx.x;
	int gridIndex = threadIdx.x;*/

	if (i >= numParticles * 27) return;
	uint index = i / 27;
	uint gridIndex = i % 27;

	uint originalIndex = gridParticleIndex[index];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	int x = (gridIndex % 3) - 1;
	int y = ((gridIndex / 3) % 3) - 1;
	int z = (gridIndex / (3 * 3)) - 1;

	int3 neighbourCellPos = gridPos + make_int3(x, y, z);

	initLink2(gridParticleIndex, neighbourCellPos, index,
							originalIndex, oldPos, oldPositions, cellStart, cellEnd,
							neighbours, numNeighb);
}

__global__
void externalForceInteractionD(uint* externalPariclesIndex,
		double4 *externalForceNorms, double4 *accelerations,
		double4 * colorLinks, double4 * colorDeform, double4 * colorAccel,
		uint numExternalForceParticle) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numExternalForceParticle)
		return;

	uint particleIndex = externalPariclesIndex[index];
	double3 normForce = make_double3(externalForceNorms[index]);

	if (colorLinks) {
		colorLinks[particleIndex].x = 0.0;
		colorLinks[particleIndex].y = 0.0;
		colorLinks[particleIndex].z = 1.0;

		colorDeform[particleIndex].x = 0.0;
		colorDeform[particleIndex].y = 0.0;
		colorDeform[particleIndex].z = 1.0;

		colorAccel[particleIndex].x = 0.0;
		colorAccel[particleIndex].y = 0.0;
		colorAccel[particleIndex].z = 1.0;

		colorAccel[particleIndex].w = 0.1;
		colorDeform[particleIndex].w = 0.1;
		colorLinks[particleIndex].w = 0.1;
	}

	if (!length(normForce)) {
		accelerations[particleIndex].x = 0;
		accelerations[particleIndex].y = 0;
		accelerations[particleIndex].z = 0;
		return;
	}

#if SOFTFVAL

	if (params.externalDeltaForce <= 1000) {
		accelerations[particleIndex] += externalForceNorms[index]
				* (params.externalDeltaForce / params.automateMass);
	}
#else

	accelerations[particleIndex] += externalForceNorms[index]
					* (params.externalDeltaForce / params.automateMass);
#endif


}

__global__
void updateSelectedParticlesD(uint* selectedParticlesIndices,
		double4 * colorLinks, double4 * colorDeform, double4 * colorAccel,
		uint numSelectedParticles) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numSelectedParticles)
		return;
	uint particleIndex = selectedParticlesIndices[index];

	colorLinks[particleIndex].x = 0.0;
	colorLinks[particleIndex].y = 1.0;
	colorLinks[particleIndex].z = 0.0;

	colorDeform[particleIndex].x = 0.0;
	colorDeform[particleIndex].y = 1.0;
	colorDeform[particleIndex].z = 0.0;

	colorAccel[particleIndex].x = 0.0;
	colorAccel[particleIndex].y = 1.0;
	colorAccel[particleIndex].z = 0.0;

	colorAccel[particleIndex].w = 0.6;
	colorDeform[particleIndex].w = 0.6;
	colorLinks[particleIndex].w = 0.6;
}

__global__
void collideChangedD(
		double *stress,
		double4 *velocity,					//input: velocity
		double4 *accelerations,               // output: new accel
		double4 *oldPositions,               // input: sorted positions
		double4 *prevPos, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, double *deformation,	// input: average deformation
		double4 *colorAccel, //output: color-accel
		uint *gridParticleIndex,    // input: sorted particle indices
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb, //input: neighbors of particle
		uint numParticles) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles * 27)
		return;

	uint i = index / 27;

	uint originalIndex = gridParticleIndex[i];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	int z = index % 3 - 1;
	int y = (index / 3) % 3 - 1;
	int x = (index / (3 * 3)) % 3 - 1;

	// examine neighbouring cells
	double3 force = make_double3(0.0);
	angAccelerations[originalIndex] = make_double4(0.0, 0.0, 0.0, 0.0);
	stress[originalIndex] = 0.0;

	int3 neighbourPos = gridPos + make_int3(x, y, z);
	force += collideCellStress(gridParticleIndex, neighbourPos, i,
			originalIndex, oldPositions, prevPos, velocity, angles, prevAngles,
			angAccelerations, cellStart, cellEnd, neighbrs, numNeighb,
			deformation, stress);

	if (numNeighb[originalIndex])
		stress[originalIndex] = stress[originalIndex] / numNeighb[originalIndex];

	// collide with cursor sphere
	force += collideCollider(oldPos, params.colliderPos,
			make_double3(velocity[originalIndex]), make_double3(0.0, 0.0, 0.0),
			params.particleRadius, params.colliderRadius, 0.0);

	double tension = length(force);

	if (colorAccel) {
		colorAccel[originalIndex].x = 1.0;
		colorAccel[originalIndex].y = 1.0
				- min(tension / (2 * MAX_TENSION_COLOR), 1.0);
		colorAccel[originalIndex].z = 1.0
				- min(tension / (MAX_TENSION_COLOR), 1.0);
		colorAccel[originalIndex].w = 0.5;
	}

	accelerations[originalIndex] += make_double4(force / params.automateMass, 0.0);
}

__global__
void collideD(
		double *stress,
		double4 *velocity,					//input: velocity
		double4 *accelerations,               // output: new accel
		double4 *oldPositions,               // input: sorted positions
		double4 *prevPos, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, double *deformation,	// input: average deformation
		double4 *colorAccel, //output: color-accel
		uint *gridParticleIndex,    // input: sorted particle indices
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb, //input: neighbors of particle
		uint numParticles) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;
	uint originalIndex = gridParticleIndex[index];

	double3 oldPos = make_double3(oldPositions[originalIndex]);

	// get address in grid
	int3 gridPos = calcGridPos(oldPos);

	// examine neighbouring cells
	double3 force = make_double3(0.0);
	angAccelerations[originalIndex] = make_double4(0.0, 0.0, 0.0, 0.0);
	stress[originalIndex] = 0.0;

	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				force += collideCellStress(gridParticleIndex, neighbourPos, index,
										originalIndex, oldPositions, prevPos, velocity, angles,
										prevAngles, angAccelerations, cellStart, cellEnd,
										neighbrs, numNeighb, deformation, stress);
				/*force += collideCell(gridParticleIndex, neighbourPos, index,
						originalIndex, oldPositions, prevPos, velocity, angles,
						prevAngles, angAccelerations, cellStart, cellEnd,
						neighbrs, numNeighb, deformation);*/
			}
		}
	}

	if (numNeighb[originalIndex])
		stress[originalIndex] = stress[originalIndex] / numNeighb[originalIndex];

	// collide with cursor sphere
	force += collideCollider(oldPos, params.colliderPos,
			make_double3(velocity[originalIndex]), make_double3(0.0, 0.0, 0.0),
			params.particleRadius, params.colliderRadius, 0.0);

	double tension = length(force);

	if (colorAccel) {
		colorAccel[originalIndex].x = 1.0;
		colorAccel[originalIndex].y = 1.0
				- min(tension / (2 * MAX_TENSION_COLOR), 1.0);
		colorAccel[originalIndex].z = 1.0
				- min(tension / (MAX_TENSION_COLOR), 1.0);
		colorAccel[originalIndex].w = 0.5;
	}

	accelerations[originalIndex] += make_double4(force / params.automateMass, 0.0);
}

__device__
double3 getClosestPoint(double3 rayBegin, double3 rayEnd, double3 pos) {
	double3 ray = rayEnd - rayBegin;
	double lenSqrRay = dot(ray, ray);
	double3 vecPosMinusRayBegin = pos - rayBegin;
	double dotVecAndPoint = dot(vecPosMinusRayBegin, ray);
	double projectionParamVecOntoPoint = dotVecAndPoint / lenSqrRay;
	return rayBegin + ray * projectionParamVecOntoPoint;
}

__global__
void intersectionParticleD(double3 rayBegin, double3 rayEnd, uint* selectedParticlesIndicies, double* distanceToCamera, double4 *positions, uint numParticles, uint* foundCollision) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	distanceToCamera[index] = length(rayEnd - rayBegin);
	selectedParticlesIndicies[index] = index;

	double3 pos = make_double3(positions[index]);
	double3 closestPoint = getClosestPoint(rayBegin, rayEnd, pos);
	double len = length(pos - closestPoint);
	if (len > params.particleRadius)
		return;
	*foundCollision = 1; // intersection exists
	distanceToCamera[index] = length(pos - rayBegin);
}


/*FUNCTIONS FOR TESTING*/

__global__
void integrateVerletFunctionTestD(double4 *posData, double4 *posPrevData, double4 *velData, double4 *accelData,
		double4 *thetaData,
		double4 *thetaPrevData,
		double4 *thetaAccelData, uint numParticles, uint numThreads) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numThreads)
		return;

	for (; index < numParticles; index += numThreads) {
			double3 pos = make_double3(posData[index]);
			double3 pos_prev = make_double3(posPrevData[index]);
			double3 vel = make_double3(velData[index]);
			double3 accel = make_double3(accelData[index]);

			double3 theta = make_double3(thetaData[index]);
			double3 thetaPrev = make_double3(thetaPrevData[index]);
			double3 thetaAccel = make_double3(thetaAccelData[index]);

			double3 tempPos = pos;
			double3 tempTheta = theta;

			// new position = 2 * old position -  old_prev_position + accel * deltaTime * deltaTime
			pos = 2.0 * pos - pos_prev + accel * params.timeStep * params.timeStep;

			theta = 2.0 * theta - thetaPrev
					+ thetaAccel * params.timeStep * params.timeStep;

			//vel_prev = (new_position - pre_old_position) / (2 * deltaTime)
			vel = 1.0 / (2.0 * params.timeStep) * (pos - pos_prev);

			pos_prev = tempPos;
			thetaPrev = tempTheta;

			posData[index] = make_double4(pos, posData[index].w);
			posPrevData[index] = make_double4(pos_prev, 0.0);
			velData[index] = make_double4(vel, 0.0);
			accelData[index] = make_double4(accel, 0.0);
			thetaData[index] = make_double4(theta, 0.0);
			thetaPrevData[index] = make_double4(thetaPrev, 0.0);
			thetaAccelData[index] = make_double4(thetaAccel, 0.0);
	}
}

// calculate grid hash value for each particle
__global__
void calcHashTestD(uint *gridParticleHash,  // output
		uint *gridParticleIndex, // output
		double4 *pos,               // input: positions
		uint numParticles, uint numThreads) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numThreads)
		return;

	for (; index < numParticles; index += numThreads) {
		volatile double4 p = pos[index];

		// get address in grid
		int3 gridPos = calcGridPos(make_double3(p.x, p.y, p.z));
		uint hash = calcGridHash(gridPos);

		// store grid hash and particle index
		gridParticleHash[index] = hash;
		gridParticleIndex[index] = index;
	}
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartTestD(uint *cellStart,   // output: cell start index
		uint *cellEnd,          // output: cell end index
		uint *gridParticleHash, // input: sorted grid hashes
		uint *gridParticleIndex, // input: sorted particle indices
		uint numParticles, uint numThreads) {
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numThreads)
		return;

	for (; index < numParticles; index += numThreads) {
		uint hash = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			sharedHash[0] = gridParticleHash[index - 1];
		}

		__syncthreads();

		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1) {
			cellEnd[hash] = index + 1;
		}
	}
}


__global__
void updateLinksAndDeformTestD(uint *gridParticleIndex, double * deformation,
		double4 *oldPositions, double4 *colorLinks, double4 *colorDeform,
		uint *cellStart, uint *cellEnd, uint* neighbours, uint* numNeighb,
		uint numParticles, uint numThreads) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numThreads)
		return;

	for (; index < numParticles; index += numThreads) {
		uint originalIndex = gridParticleIndex[index];

		double3 oldPos = make_double3(oldPositions[originalIndex]);

		// get address in grid
		int3 gridPos = calcGridPos(oldPos);

		for (int z = -1; z <= 1; z++)
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					int3 neighbourCellPos = gridPos + make_int3(x, y, z);
					updateLink(gridParticleIndex, neighbourCellPos, index,
							originalIndex, oldPos, oldPositions, cellStart,
							cellEnd, neighbours, numNeighb, deformation);
				}
			}

		if (numNeighb[originalIndex]) {
			deformation[originalIndex] = deformation[originalIndex]
					/ numNeighb[originalIndex];
		}

		//use GL
		if (colorDeform) {
			double k1 = 1.0 - abs(deformation[originalIndex] * 50.0);
			colorDeform[originalIndex].x = 1.0;
			colorDeform[originalIndex].y = k1;
			colorDeform[originalIndex].z = k1;
			colorDeform[originalIndex].w = 0.5;

			double k2 = (double) numNeighb[originalIndex]
					/ (double) params.mcaMaxNumNeighbors;
			colorLinks[originalIndex].x = k2;
			colorLinks[originalIndex].y = k2;
			colorLinks[originalIndex].z = k2;
			colorLinks[originalIndex].w = 0.5;
		}
	}
}

__global__
void externalForceInteractionTestD(uint* externalPariclesIndex,
		double4 *externalForceNorms, double4 *accelerations,
		double4 * colorLinks, double4 * colorDeform, double4 * colorAccel,
		uint numExternalForceParticle, uint numThreads) {

	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numExternalForceParticle)
		return;

	for (; index < numExternalForceParticle; index += numThreads) {
		uint particleIndex = externalPariclesIndex[index];
		double3 normForce = make_double3(externalForceNorms[index]);

		// use GL
		if (colorLinks) {
			colorLinks[particleIndex].x = 0.0;
			colorLinks[particleIndex].y = 0.0;
			colorLinks[particleIndex].z = 1.0;

			colorDeform[particleIndex].x = 0.0;
			colorDeform[particleIndex].y = 0.0;
			colorDeform[particleIndex].z = 1.0;

			colorAccel[particleIndex].x = 0.0;
			colorAccel[particleIndex].y = 0.0;
			colorAccel[particleIndex].z = 1.0;

			colorAccel[particleIndex].w = 0.1;
			colorDeform[particleIndex].w = 0.1;
			colorLinks[particleIndex].w = 0.1;
		}

		if (!length(normForce)) {
			accelerations[particleIndex].x = 0;
			accelerations[particleIndex].y = 0;
			accelerations[particleIndex].z = 0;
			return;
		}

		accelerations[particleIndex] += externalForceNorms[index]
							* (params.externalDeltaForce / params.automateMass);
	}
}

__global__
void collideTestD(
		double4 *velocity,					//input: velocity
		double4 *accelerations,               // output: new accel
		double4 *oldPositions,               // input: sorted positions
		double4 *prevPos, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, double *deformation,	// input: average deformation
		double4 *colorAccel, //output: color-accel
		uint *gridParticleIndex,    // input: sorted particle indices
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb, //input: neighbors of particle
		uint numParticles, uint numThreads) {
	uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numThreads)
		return;

	for (; index < numParticles; index += numThreads) {
		uint originalIndex = gridParticleIndex[index];

		// read particle data from sorted arrays
		double3 oldPos = make_double3(oldPositions[originalIndex]);

		// get address in grid
		int3 gridPos = calcGridPos(oldPos);

		// examine neighbouring cells
		double3 force = make_double3(0.0);
		angAccelerations[originalIndex] = make_double4(0.0, 0.0, 0.0, 0.0);

		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					/*force += collideCell(gridParticleIndex, neighbourPos, index,
							originalIndex, oldPositions, prevPos, velocity,
							angles, prevAngles, angAccelerations, cellStart,
							cellEnd, neighbrs, numNeighb, deformation);*/
				}
			}
		}

		// collide with cursor sphere
		force += collideCollider(oldPos, params.colliderPos,
				make_double3(velocity[originalIndex]),
				make_double3(0.0, 0.0, 0.0), params.particleRadius,
				params.colliderRadius, 0.0);

		//bUseGL == true
		if (colorAccel) {
			double tension = length(force);

			colorAccel[originalIndex].x = 1.0;
			colorAccel[originalIndex].y = 1.0
					- min(tension / (2 * MAX_TENSION_COLOR), 1.0);
			colorAccel[originalIndex].z = 1.0
					- min(tension / (MAX_TENSION_COLOR), 1.0);
			colorAccel[originalIndex].w = 0.5;
		}

		accelerations[originalIndex] = make_double4(force / params.automateMass, 0.0);
	}
}

#endif
