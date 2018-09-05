/*
 * particles_kernel_impl_cpu.h
 *
 *  Created on: Sep 29, 2017
 *      Author: myvaheed
 */

#ifndef PARTICLES_KERNEL_IMPL_CPU_H_
#define PARTICLES_KERNEL_IMPL_CPU_H_

#include <omp.h>
#include "../tools/helper_double3_cpu.h"
#include <cmath>

static McaParams params;

void setParametersCPU(McaParams *hostParams) {
	params = *hostParams;
}
void integrateVerletSystemCPU(double *pos, double *pos_prev, double *vel,
		double *accel, double *theta, double *thetaPrev, double *thetaAccel,
		uint numParticles) {
	double4* posData = (double4*) pos;
	double4* posPrevData = (double4*) pos_prev;
	double4* velData = (double4*) vel;
	double4* accelData = (double4*) accel;
	double4* thetaData = (double4*) theta;
	double4* thetaPrevData = (double4*) thetaPrev;
	double4* thetaAccelData = (double4*) thetaAccel;

#pragma omp parallel for
	for (int index = 0; index < numParticles; index++) {
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
}

int3 calcGridPosCPU(double3 p) {
	int3 gridPos;
	gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
	gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
	gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
uint calcGridHashCPU(int3 gridPos) {
	return (gridPos.x * 73856093 + gridPos.y * 19349663 + gridPos.z * 83492791)
			% params.numGridCells;
}

void calcHashCPU(uint *gridParticleHash, uint *gridParticleIndex, double *pos,
		int numParticles) {
#pragma omp parallel for
	for (int index = 0; index < numParticles; index++) {
		/*volatile */double4 p = ((double4*) pos)[index];

		// get address in grid
		int3 gridPos = calcGridPosCPU(make_double3(p.x, p.y, p.z));
		uint hash = calcGridHashCPU(gridPos);

		// store grid hash and particle index
		gridParticleHash[index] = hash;
		gridParticleIndex[index] = index;
	}
}

void reorderDataAndFindCellStartCPU(uint *cellStart, uint *cellEnd,
		uint *gridParticleHash, uint *gridParticleIndex, uint numParticles,
		uint numCells) {
	memset(cellStart, 0xffffffff, numCells * sizeof(uint));
	uint sharedHash[omp_get_num_threads() + 1];

#pragma omp parallel for /*private(sharedHash)*/ shared(sharedHash)
	for (int i = 0; i < numParticles; i++) {
		int tid = omp_get_thread_num();
		uint hash = gridParticleHash[i];
		sharedHash[tid + 1] = hash;

		if (i > 0 && tid == 0) {
			sharedHash[0] = gridParticleHash[i - 1];
		}
	}


		//printf("hash %d tid = %d, num_threads = %d\n", hash, tid, omp_get_num_threads());
#pragma omp parallel for /*private(sharedHash)*/ shared(sharedHash)
	for (int i = 0; i < numParticles; i++) {
		int tid = omp_get_thread_num();
		uint hash = gridParticleHash[i];
		if (i == 0 || hash != sharedHash[tid]) {
			cellStart[hash] = i;

			if (i > 0)
				cellEnd[sharedHash[tid]] = i;
		}

		if (i == numParticles - 1) {
			cellEnd[hash] = i + 1;
		}
	}
}


inline bool isLinked(uint* neighbours, uint* numNeighb, uint index, uint neighbIndex, int& indexInNeighbrs) {

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
					neighbours[index * params.mcaMaxNumNeighbors
							+ --numNeighb[index]];

			return;
		}

		deformation[index] += (dist - params.automateD) / params.automateD;
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

			deformation[index] += (dist - params.automateD) / params.automateD;
			return;
		}
	}
}

void updateLinksAndDeformCPU(uint *gridParticleIndex, double *deformation,
		double *oldPositions, double *colorLinks, double *colorDeform,
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb,
		uint numParticles) {
	uint numThreads, numBlocks;

	// set all data to zero
	memset(deformation, 0, numParticles * sizeof(double));

	if (colorLinks) {
		memset(colorLinks, 0, 4 * numParticles * sizeof(double));
		memset(colorDeform, 0, 4 * numParticles * sizeof(double));
	}


#pragma omp parallel for
	for (int i = 0; i < numParticles; i++) {
		uint originalIndex = gridParticleIndex[i];

		double3 oldPos;
		oldPos.x = oldPositions[4 * originalIndex];
		oldPos.y = oldPositions[4 * originalIndex + 1];
		oldPos.z = oldPositions[4 * originalIndex + 2];

		// get address in grid
		int3 gridPos = calcGridPosCPU(oldPos);

		for (int z = -1; z <= 1; z++)
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					int3 index = make_int3(x, y, z);
					int3 neighbourCellPos;
					neighbourCellPos.x = gridPos.x + index.x;
					neighbourCellPos.y = gridPos.y + index.y;
					neighbourCellPos.z = gridPos.z + index.z;

					uint gridHash = calcGridHashCPU(neighbourCellPos);

					uint startIndex = cellStart[gridHash];

					if (startIndex != 0xffffffff)          // cell is not empty
							{
						uint endIndex = cellEnd[gridHash];
						for (uint j = startIndex; j < endIndex; j++) {
							if (j != i) // check not colliding with self
									{
								uint originalIndexNeighb = gridParticleIndex[j];
								double3 oldPos2;
								oldPos2.x = oldPositions[4 * originalIndexNeighb];
								oldPos2.y =
										oldPositions[4 * originalIndexNeighb + 1];
								oldPos2.z =
										oldPositions[4 * originalIndexNeighb + 2];

								checkLink(neighbrs, numNeighb, originalIndex,
										originalIndexNeighb, oldPos, oldPos2,
										deformation);
							}
						}
					}
				}
			}

		if (numNeighb[originalIndex]) {
			deformation[originalIndex] = deformation[originalIndex]
					/ numNeighb[originalIndex];
		}

		//bUseGL == true
		if (colorDeform) {
			double k1 = 1.0 - abs(deformation[originalIndex] * 50.0);
			colorDeform[4 * originalIndex] = 1.0;
			colorDeform[4 * originalIndex + 1] = k1;
			colorDeform[4 * originalIndex + 2] = k1;
			colorDeform[4 * originalIndex + 2] = 0.5;

			double k2 = (double) numNeighb[originalIndex]
					/ (double) params.mcaMaxNumNeighbors;
			colorLinks[4 * originalIndex] = k2;
			colorLinks[4 * originalIndex + 1] = k2;
			colorLinks[4 * originalIndex + 2] = k2;
			colorLinks[4 * originalIndex + 3] = 0.5;
		}

	}
}

void externalForceInteractionCPU(uint* externalParticleIndex,
		double* extForceNorms, double *accels, double *colorLinks,
		double *colorDeform, double *colorAccel, uint numExternalParticles) {
	double4* externalForceNorms = (double4*) extForceNorms;
	double4* accelerations = (double4*) accels;

#pragma omp parallel for
	for (int i = 0; i < numExternalParticles; i++) {
		uint particleIndex = externalParticleIndex[i];

		//bUseGL == true
		if (colorLinks) {
			colorLinks[4 * particleIndex] = 0.0;
			colorLinks[4 * particleIndex + 1] = 0.0;
			colorLinks[4 * particleIndex + 2] = 1.0;

			colorDeform[4 * particleIndex] = 0.0;
			colorDeform[4 * particleIndex + 1] = 0.0;
			colorDeform[4 * particleIndex + 2] = 1.0;

			colorAccel[4 * particleIndex] = 0.0;
			colorAccel[4 * particleIndex + 1] = 0.0;
			colorAccel[4 * particleIndex + 2] = 1.0;

			colorAccel[4 * particleIndex + 3] = 0.1;
			colorDeform[4 * particleIndex + 3] = 0.1;
			colorLinks[4 * particleIndex + 3] = 0.1;
		}
		if (!length(make_double3(externalForceNorms[i]))) {
			accelerations[particleIndex].x = 0;
			accelerations[particleIndex].y = 0;
			accelerations[particleIndex].z = 0;
		} else {
			accelerations[particleIndex] += externalForceNorms[i]
							* params.externalDeltaForce;
		}
	}
}


inline double3 getNeighbrsNormForce(uint index, uint neighbIndex, double4* positions,
		double *deformations, uint* neighbrs, uint* numNeighbrs, double3 norm) {
	double3 force = make_double3(0.0, 0.0, 0.0);
	double3 posB;
	posB.x = positions[index].x;
	posB.y = positions[index].y;
	posB.z = positions[index].z;

	for (int i = 0; i < numNeighbrs[index]; i++) {
		if (neighbIndex == neighbrs[index * params.mcaMaxNumNeighbors + i])
			continue;
		double3 posA;
		posA.x = positions[neighbrs[index * params.mcaMaxNumNeighbors + i]].x;
		posA.y = positions[neighbrs[index * params.mcaMaxNumNeighbors + i]].y;
		posA.z = positions[neighbrs[index * params.mcaMaxNumNeighbors + i]].z;

		double3 relPos;
		relPos.x = posA.x - posB.x;
		relPos.y = posA.y - posB.y;
		relPos.z = posA.z - posB.z;

		double dist = length(relPos);
		double3 norm2;
		norm2.x = relPos.x / dist;
		norm2.y = relPos.y / dist;
		norm2.z = relPos.z / dist;

		double q = dist / 2.0;
		double S = params.automateS;
		double deformation = (q - (params.automateD / 2.0))
				/ (params.automateD / 2.0);
		force.x += (2.0 * params.mcaG * (deformation - deformations[index])) * S
				* dot(norm, norm2) * norm.x;
		force.y += (2.0 * params.mcaG * (deformation - deformations[index])) * S
						* dot(norm, norm2) * norm.y;
		force.z += (2.0 * params.mcaG * (deformation - deformations[index])) * S
						* dot(norm, norm2) * norm.z;
	}

	posB.x = positions[neighbIndex].x;
	posB.y = positions[neighbIndex].y;
	posB.z = positions[neighbIndex].z;

	for (int i = 0; i < numNeighbrs[neighbIndex]; i++) {
		if (neighbIndex == neighbrs[neighbIndex * params.mcaMaxNumNeighbors + i])
			continue;
		double3 posA;
		posA.x = positions[neighbrs[neighbIndex * params.mcaMaxNumNeighbors + i]].x;
		posA.y = positions[neighbrs[neighbIndex * params.mcaMaxNumNeighbors + i]].y;
		posA.z = positions[neighbrs[neighbIndex * params.mcaMaxNumNeighbors + i]].z;

		double3 relPos;
		relPos.x = posA.x - posB.x;
		relPos.y = posA.y - posB.y;
		relPos.z = posA.z - posB.z;

		double dist = length(relPos);
		double3 norm2;
		norm2.x = relPos.x / dist;
		norm2.y = relPos.y / dist;
		norm2.z = relPos.z / dist;

		double q = dist / 2.0;
		double S = params.automateS;
		double deformation = (q - (params.automateD / 2.0))
				/ (params.automateD / 2.0);
		force.x += (2.0 * params.mcaG * (deformation - deformations[neighbIndex])) * S
				* dot(norm, norm2) * norm.x;
		force.y += (2.0 * params.mcaG * (deformation - deformations[neighbIndex])) * S
				* dot(norm, norm2) * norm.y;
		force.z += (2.0 * params.mcaG * (deformation - deformations[neighbIndex])) * S
				* dot(norm, norm2) * norm.z;
	}

	return force;
}


inline double3 mcaSpheres(uint index, uint neighbIndex, double4 *positions,
		double4 *prevPositions, double4 *velocities, double4 *angles,
		double4 *prevAngles, double4 *angAccelerations, double *deformations,
		uint* neighbrs, uint* numNeighb) {

	double3 posA = make_double3(positions[index].x, positions[index].y, positions[index].z);
	double3 posB = make_double3(positions[neighbIndex].x, positions[neighbIndex].y, positions[neighbIndex].z);
	double3 prevPosA = make_double3(prevPositions[index].x, prevPositions[index].y, prevPositions[index].z);
	double3 prevPosB = make_double3(prevPositions[neighbIndex].x, prevPositions[neighbIndex].y, prevPositions[neighbIndex].z);
	double3 velA = make_double3(velocities[index].x, velocities[index].y, velocities[index].z);
	double3 velB = make_double3(velocities[neighbIndex].x, velocities[neighbIndex].y, velocities[neighbIndex].z);
	double3 angleA = make_double3(angles[index].x, angles[index].y, angles[index].z);
	double3 angleB = make_double3(angles[neighbIndex].x, angles[neighbIndex].y, angles[neighbIndex].z);
	double3 prevAngleA = make_double3(prevAngles[index].x, prevAngles[index].y, prevAngles[index].z);
	double3 prevAngleB = make_double3(prevAngles[neighbIndex].x, prevAngles[neighbIndex].y, prevAngles[neighbIndex].z);

	// calculate relative position
	double3 relPos;
	relPos.x = posB.x - posA.x;
	relPos.y = posB.y - posA.y;
	relPos.z = posB.z - posA.z;

	double dist = length(relPos);

	double q = dist / 2.0;
	double S = params.automateS;
	/*double S;
	if (q > params.particleRadius) {
		S = 0;
	} else {
		//S = 4 * (params.particleRadius * params.particleRadius - q * q);
		S = params.automateD * 2
				* sqrt(params.particleRadius * params.particleRadius - q * q);
	}*/
	double3 accel = make_double3(0.0, 0.0, 0.0);
	double deformation = (q - (params.automateD / 2.0))
			/ (params.automateD / 2.0);

	double3 norm;
	norm.x = relPos.x / dist;
	norm.y = relPos.y / dist;
	norm.z = relPos.z / dist;

	double3 prevRelPos;
	prevRelPos.x = prevPosB.x - prevPosA.x;
	prevRelPos.y = prevPosB.y - prevPosA.y;
	prevRelPos.z = prevPosB.z - prevPosA.z;

	double3 prevNorm;
	double lenPrevRelPos = length(prevRelPos);
	prevNorm.x = prevRelPos.x / lenPrevRelPos;
	prevNorm.y = prevRelPos.y / lenPrevRelPos;
	prevNorm.z = prevRelPos.z / lenPrevRelPos;

	// relative velocity
	double3 relVel;
	relVel.x = velB.x - velA.x;
	relVel.y = velB.y - velA.y;
	relVel.z = velB.z - velA.z;

	// relative tangential velocity
	double3 tanVel;
	tanVel.x = relVel.x - (dot(relVel, norm) * norm.x);
	tanVel.y = relVel.y - (dot(relVel, norm) * norm.y);
	tanVel.z = relVel.z - (dot(relVel, norm) * norm.z);

	//NORMAL ACCELERATIONS
	double3 neighbrsNormForces = getNeighbrsNormForce(index, neighbIndex,
			positions, deformations, neighbrs, numNeighb, norm);

	double forceP = (2.0 * params.mcaG * (deformation - deformations[index]))
			* S;
	accel.x += forceP * norm.x + neighbrsNormForces.x;
	accel.y += forceP * norm.y + neighbrsNormForces.y;
	accel.z += forceP * norm.z + neighbrsNormForces.z;

	double forcePVolume = -(params.mcaK * params.mcaD * -deformations[index])
			* S;
	accel.x += forcePVolume * norm.x;
	accel.y += forcePVolume * norm.y;
	accel.z += forcePVolume * norm.z;

	//TANGETIAL ACCELERATIONS
	// relative angle of pair
	double3 relThetaOfPair = acos(dot(norm, prevNorm))
			/ sqrt(1.0 - dot(norm, prevNorm) * dot(norm, prevNorm))
			* cross(norm, prevNorm);
	double3 relThetaA = angleA - prevAngleA;
	double3 relThetaB = angleB - prevAngleB;

	if (std::isnan(relThetaOfPair.x))
		accel += -1.0
				* (params.mcaG * (-1.0 * cross(relThetaA, norm))
						+ params.mcaG * (-1.0 * cross(relThetaB, norm)));
	else
		accel += -1.0
				* (params.mcaG * (cross(relThetaOfPair - relThetaA, norm))
						+ params.mcaG
								* (cross(relThetaOfPair - relThetaB, norm)));

	// relative tangential accel
	double3 tanAccel = accel - (dot(accel, norm) * norm);
	angAccelerations[index] += make_double4(cross(tanAccel, norm) / q, 0);

	return accel;
}

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
		double3 tanVel = relVel - (dot(relVel, norm) * norm);

		// tangential shear force
		accel += params.mcaShear * tanVel;

		// spring force
		accel += norm * 0.5f
				* ((1.0 + params.bounceCollision) * dot(relVel, norm)
						+ (collideDist - dist)) / params.timeStep;
	}

	return accel;
}

inline double3 collideCollider(double3 posA, double3 posB, double3 velA, double3 velB,
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

inline double3 collideCell(uint* gridParticlesIndex, int3 gridPos, uint sortedIndex,
		uint originalIndex, double4 *oldPositions, double4 *prevPositions,
		double4 *velocity, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, uint *cellStart, uint *cellEnd,
		uint* neighbrs, uint* numNeighb, double* deformation) {
	uint gridHash = calcGridHashCPU(gridPos);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	double3 force = make_double3(0.0, 0.0, 0.0);

	if (startIndex != 0xffffffff)          // cell is not empty
			{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++) {
			if (j != sortedIndex)               // check not colliding with self
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
							neighbrs, numNeighb);

				} else {
					//printf("unlinked\n");
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

inline void collideImplCPU(
		double4 *velocity,					//input: velocity
		double4 *accelerations,               // output: new accel
		double4 *oldPositions,               // input: sorted positions
		double4 *prevPos, double4 *angles, double4 *prevAngles,
		double4 *angAccelerations, double *deformation,	// input: average deformation
		double4 *colorAccel, //output: color-accel
		uint *gridParticleIndex,    // input: sorted particle indices
		uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb, //input: neighbors of particle
		uint numParticles) {
#pragma omp parallel for
	for (int index = 0; index < numParticles; index++) {
		uint originalIndex = gridParticleIndex[index];

		// read particle data from sorted arrays
		double3 oldPos = make_double3(oldPositions[originalIndex]);

		// get address in grid
		int3 gridPos = calcGridPosCPU(oldPos);

		// examine neighbouring cells
		double3 force = make_double3(0.0);
		angAccelerations[originalIndex] = make_double4(0.0, 0.0, 0.0, 0.0);

		for (int z = -1; z <= 1; z++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					int3 indexCell = make_int3(x, y, z);
					int3 neighbourCellPos;
					neighbourCellPos.x = gridPos.x + indexCell.x;
					neighbourCellPos.y = gridPos.y + indexCell.y;
					neighbourCellPos.z = gridPos.z + indexCell.z;
					force += collideCell(gridParticleIndex, neighbourCellPos, index,
							originalIndex, oldPositions, prevPos, velocity,
							angles, prevAngles, angAccelerations, cellStart,
							cellEnd, neighbrs, numNeighb, deformation);
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
					- std::min(tension / (2 * MAX_TENSION_COLOR), 1.0);
			colorAccel[originalIndex].z = 1.0
					- std::min(tension / (MAX_TENSION_COLOR), 1.0);
			colorAccel[originalIndex].w = 0.5;
		}


		accelerations[originalIndex] = make_double4(force, 0.0);
	}
}

void collideCPU(double *vel, double *newAccel, double *sortedPos,
		double *prevPos, double *angles, double *prevAngles,
		double *angAccelerations, double *deformation, double *colorAccel,
		uint *gridParticleIndex, uint *cellStart, uint *cellEnd, uint* neighbrs,
		uint* numNeighb, uint numParticles) {

	collideImplCPU((double4 *) vel, (double4 *) newAccel, (double4 *) sortedPos,
			(double4 *) prevPos, (double4 *) angles, (double4 *) prevAngles,
			(double4 *) angAccelerations, deformation, (double4 *) colorAccel,
			gridParticleIndex, cellStart, cellEnd, neighbrs, numNeighb,
			numParticles);
}



#endif /* PARTICLES_KERNEL_IMPL_CPU_H_ */
