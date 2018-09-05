#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "McaParams.h"
#include "vector_functions.h"

// Particle system class
class McaSystem {
public:
	McaSystem(uint numParticles, double particleRadius, double amass, double space_width,
			double space_height, double space_depth, bool bUseGL);
	~McaSystem();

	enum ParticleConfig {
		CONFIG_RANDOM = 0, CONFIG_GRID, CONFIG_SQUARE, CONFIG_FCC, CONFIG_HCP, _NUM_CONFIGS
	};

	enum ParticleExternalForce {
		EXTERNAL_FORCE_DEFAULT = 0,
		EXTERNAL_FORCE_CORNER,
		EXTERNAL_FORCE_TWO_CORNERS,
		EXTERNAL_FORCE_FORCER,
		EXTERNAL_FORCE_PUSH_APART,
		EXTERNAL_FORCE_PULL_APART,
	};

	enum ParticleArray {
		POSITION = 0,
		PRE_POSITION,
		ANGLE,
		VELOCITY,
		ANGLE_PREV,
		ANGLE_ACCEL,
		ACCEL,
		DEFORM,
		EXTERNAL_FORCE_NORM,
		EXTERNAL_PARTICLES,
		SELECTED_PARTICLES_INDICES,
		GRID_INDEX,
		GRID_HASH,
		NEIGHBORS,
		STRESS,
		NUM_NEIGHBRS
	};

	void updateVelretTestMode(bool isGPU, int numThreads);
	void updateVerlet();
	void selectParticle(double3 rayOrigin, double3 rayEnd);

	void configSystem(uint numColumn, uint numRow, ParticleConfig config, ParticleExternalForce mode);
	void configSystem(uint numColumn, uint numRow, ParticleConfig config);

	void *getArray(ParticleArray array);
	void setArray(ParticleArray array, const void *data, int start, int count);

	int getNumParticles() const {
		return m_numParticles;
	}

	void writeSelectedArrayToFile(const char* file, ParticleArray arrayTime);

	int getNumExternalParticles() const {
		return m_numExternalParticles;
	}

	unsigned int getPositionBuffer() const {
		return m_posVbo;
	}
	unsigned int getColorLinksBuffer() const {
		return m_colorLinksVbo;
	}
	unsigned int getColorDeformBuffer() const {
		return m_colorDeformVbo;
	}
	unsigned int getColorAccelBuffer() const {
		return m_colorAccelVbo;
	}

	void dumpGrid();
	void dumpParticles(uint start, uint count);

	void setMcaConstModulus(double rMin, double rMax, double youngModul, double poissionCoff, bool isBrittle) {
		m_params.mcaRmax = rMax;
		m_params.mcaRmin = rMin;
		m_params.automateE = youngModul;
		m_params.automateMu = poissionCoff;
		m_params.mcaIsBrittle = isBrittle;
		m_params.mcaK = m_params.automateE / (3 * (1 - 2 * m_params.automateMu));
		m_params.mcaG = m_params.automateE / (2 * (1 + m_params.automateMu));
	}

	void setExternalForceVal(double);

	void setGravity(double x) {
		m_params.gravity = make_double3(0.0f, x, 0.0f);
	}

	void setTimestep(double timestep) {
		m_params.timeStep = timestep;
	}



	void setCollideSpring(double x) {
		m_params.spring = x;
	}
	void setCollideShear(double x) {
		m_params.mcaShear = x;
	}

	void setColliderPos(double3 x) {
		m_params.colliderPos = x;
	}
	double getParticleRadius() {
		return m_params.particleRadius;
	}
	double3 getColliderPos() {
		return m_params.colliderPos;
	}
	double getColliderRadius() {
		return m_params.colliderRadius;
	}
	uint3 getGridSize() {
		return m_params.gridSize;
	}
	double3 getWorldOrigin() {
		return m_params.worldOrigin;
	}
	double3 getCellSize() {
		return m_params.cellSize;
	}
	void setExternalForceMode(uint numColumn, uint numRow, ParticleExternalForce mode);

protected:
	// methods
	uint createVBO(uint size);
	double* mapVBO(uint vbo);
	void unmapVBO(uint vbo);

	void setExternalForceMode2D(uint numColumn, ParticleExternalForce mode);
	void setExternalForceMode3D(uint numColumn, uint numRow, ParticleExternalForce mode);
	void _initialize(int numParticles);
	void _finalize();
	void resetData();
	void initGrid(uint numColumn);
	void initSquare(uint numColumn);
	void initHCP(uint numColumn, uint numRow);

protected:
	// data
	bool m_bInitialized; // object inited
	uint m_numParticles;
	uint m_numExternalParticles;
	uint m_numSelectedParticles;

	// CPU data
	double *m_hPos;              // particle positions
	double *m_hPos_prev;            // previous particle positions method Verlet
	double *m_hAngle;  			// particle angles
	double *m_hAngle_prev;
	double *m_hAngleAccel;
	double *m_hAccel;
	double *m_hVel;  // particle velocities
	double *m_hDeform;

	double *m_hStress; //temp

	uint* m_hNeighbors; // neighbors of particle
	uint* m_hNumNeighb; // num of neighbors of particle

	uint* m_hExternalParticlesIndices; //for external forces
	double *m_hExternalForceNorms;

	uint *m_hGridParticleHash;
	uint *m_hGridParticleIndex;
	uint *m_hCellStart;
	uint *m_hCellEnd;

	uint *m_hSelectedParticleIndices;

	// GPU data
	double *m_dPos;
	double *m_dPos_prev;
	double *m_dAccel;
	double *m_dVel;
	double *m_dDeform;
	double *m_dAngle;
	double *m_dAngle_prev;
	double *m_dAngleAccel;

	double *m_dStress; //temp

	uint* m_dNeighbors; // neighbors of particle
	uint* m_dNumNeighb;

	uint* m_dExternalParticlesIndices; //for external forces
	double *m_dExternalForceNorms;

	uint *m_dSelectedParticleIndices;

	// grid data for sorting method
	uint *m_dGridParticleHash; // grid hash value for each particle
	uint *m_dGridParticleIndex; // particle index for each particle
	uint *m_dCellStart;        // index of start of each cell in sorted list
	uint *m_dCellEnd;          // index of end of cell

	bool bUseGL;

	uint m_posVbo;            // vertex buffer object for particle positions
	uint m_colorLinksVbo;          // vertex buffer object for colors-links
	uint m_colorDeformVbo; 	 // vertex buffer object for color-deform
	uint m_colorAccelVbo;

	struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_colorlinksvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_colordeformvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource *m_cuda_coloraccelvbo_resource;

	// params
	McaParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;
	uint m_externalForceMode;
	uint m_capacitySelectedParticlesArray;

	StopWatchInterface *m_timer;
};

#endif // __PARTICLESYSTEM_H__
