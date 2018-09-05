// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "McaSystem.h"
#include "McaKernel.h"
#include "McaParams.h"

#include "McaKernel_impl_CPU.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <omp.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif


McaSystem::McaSystem(uint numParticles, double particleRadius, double mass,
		double space_width, double space_heigth, double space_depth, bool bUseGL) :
		m_bInitialized(false), m_numParticles(numParticles), m_hPos(0), m_hAccel(
				0), m_dPos(0), m_dAccel(0), m_timer(NULL), m_externalForceMode(0), m_numExternalParticles(0),
				m_hSelectedParticleIndices(0), m_numSelectedParticles(0), bUseGL(bUseGL) {

	m_gridSize.x = space_width / (2.0 * particleRadius);
	m_gridSize.y = space_heigth / (2.0 * particleRadius);
	m_gridSize.z = space_depth / (2.0 * particleRadius);

	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

	m_capacitySelectedParticlesArray = MIN_SELECTED_PARTICLES_SIZE;

	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numGridCells = m_numGridCells;

	m_params.colliderPos = make_double3(-5.0, 0.0, 0.0);
	//m_params.colliderRadius = particleRadius * 4;
	m_params.colliderRadius = particleRadius;

	m_params.worldOrigin = make_double3(0.0, 0.0, 0.0);

	m_params.spaceSize.x = space_width;
	m_params.spaceSize.y = space_heigth;
	m_params.spaceSize.z = space_width;


	//    m_params.cellSize = make_double3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
	double cellSize = particleRadius * 2.0; // cell size equal to particle diameter
	m_params.cellSize = make_double3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.mcaShear = 0.1f;
	m_params.boundaryDamping = 1.0; //[0.5..1]

	m_params.gravity = make_double3(0.0, -0.0000f, 0.0);

	m_params.particleRadius = particleRadius;
	m_params.mcaIsBrittle = true;
	m_params.automateMass = mass;
	//m_params.automateD = sqrt(3)/2.0 * particleRadius * 2;
	m_params.automateD = particleRadius * 2.0;
	//Aluminum
	//m_params.automateE = 70; // ГПа
	m_params.automateE = 70000;
	m_params.automateMu = 0.34;
	//m_params.automateE = 700;

	//Ice
	/*m_params.automateE = 0.73e10;
	m_params.automateMu = 0.3;*/
	m_params.mcaD = 2;
	m_params.mcaMaxNumNeighbors = 6;
	m_params.mcaK = m_params.automateE / (3 * (1 - 2 * m_params.automateMu));
	m_params.mcaG = m_params.automateE / (2 * (1 + m_params.automateMu));
	//m_params.mcaG = 25.5f; // Aluminum.
	//m_params.mcaG = 0.1f; // Aluminum
	m_params.automateV = m_params.automateD * m_params.automateD * m_params.automateD * sqrt(3) / 2.0;
	m_params.automateS = m_params.automateD * m_params.automateD / sqrt(3);
	m_params.mcaRmin = m_params.particleRadius * (1.0 - sqrt(3.0)/2);
	m_params.bounceCollision = 1.0; //[0..1] - 0 - абсолютно-неупругий, 1 - абсолютно-упругий


	_initialize(numParticles);
}

McaSystem::~McaSystem() {
	_finalize();
	m_numParticles = 0;
}

uint McaSystem::createVBO(uint size) {
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

double* McaSystem::mapVBO(uint vbo) {
	double* ptr;
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	ptr = (double *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return ptr;
}

void McaSystem::unmapVBO(uint vbo) {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

inline double lerp(double a, double b, double t) {
	return a + t * (b - a);
}

// create a color ramp
void colorRamp(double t, double *r) {
	const int ncolors = 7;
	double c[ncolors][3] = { { 1.0, 0.0, 0.0, }, { 1.0, 0.5, 0.0, }, { 1.0, 1.0,
			0.0, }, { 0.0, 1.0, 0.0, }, { 0.0, 1.0, 1.0, }, { 0.0, 0.0, 1.0, },
			{ 1.0, 0.0, 1.0, }, };
	t = t * (ncolors - 1);
	int i = (int) t;
	double u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
}

void McaSystem::_initialize(int numParticles) {

	assert(!m_bInitialized);

	m_numParticles = numParticles;

	// allocate host storage
	m_hPos = new double[m_numParticles * 4];
	m_hPos_prev = new double[m_numParticles * 4];
	m_hAngle = new double[m_numParticles * 4];
	m_hAngle_prev = new double[m_numParticles * 4];
	m_hAngleAccel = new double[m_numParticles * 4];
	m_hAccel = new double[m_numParticles * 4];
	m_hVel = new double[m_numParticles * 4];
	m_hDeform = new double[m_numParticles];

	m_hStress = new double[m_numParticles];

	m_hNeighbors = new uint[m_params.mcaMaxNumNeighbors * m_numParticles];
	m_hNumNeighb = new uint[m_numParticles];

	m_hExternalParticlesIndices = new uint[m_numParticles];
	m_hExternalForceNorms = new double[4 * m_numParticles];

	m_hSelectedParticleIndices = new uint[m_capacitySelectedParticlesArray];

	m_hGridParticleIndex = new uint[m_numParticles];
	m_hGridParticleHash = new uint[m_numParticles];
	m_hCellStart = new uint[m_numGridCells];
	m_hCellEnd = new uint[m_numGridCells];

	// allocate GPU data
	unsigned int memSize = sizeof(double) * 4 * m_numParticles;

	if (bUseGL) {
		m_posVbo = createVBO(memSize);
		registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

		m_colorLinksVbo = createVBO(memSize);
		registerGLBufferObject(m_colorLinksVbo, &m_cuda_colorlinksvbo_resource);

		m_colorDeformVbo = createVBO(memSize);
		registerGLBufferObject(m_colorDeformVbo,
				&m_cuda_colordeformvbo_resource);

		m_colorAccelVbo = createVBO(memSize);
		registerGLBufferObject(m_colorAccelVbo, &m_cuda_coloraccelvbo_resource);
	} else {
		allocateArray((void **) &m_dPos, memSize);
	}

	allocateArray((void **) &m_dPos_prev, memSize);
	allocateArray((void **) &m_dAccel, memSize);
	allocateArray((void **) &m_dVel, memSize);
	allocateArray((void **) &m_dDeform, sizeof(double) * m_numParticles);

	allocateArray((void **) &m_dAngle, memSize);
	allocateArray((void **) &m_dAngle_prev, memSize);
	allocateArray((void **) &m_dAngleAccel, memSize);

	allocateArray((void **) &m_dStress, sizeof(double) * m_numParticles);

	allocateArray((void **) &m_dNeighbors, m_numParticles * m_params.mcaMaxNumNeighbors * sizeof(uint));
	allocateArray((void **) &m_dNumNeighb, m_numParticles * sizeof(uint));

	allocateArray((void **) &m_dExternalParticlesIndices,
			m_numParticles * sizeof(uint));
	allocateArray((void **) &m_dExternalForceNorms, memSize);

	allocateArray((void **) &m_dSelectedParticleIndices, m_capacitySelectedParticlesArray * sizeof(uint));

	allocateArray((void **) &m_dGridParticleHash,
			m_numParticles * sizeof(uint));
	allocateArray((void **) &m_dGridParticleIndex,
			m_numParticles * sizeof(uint));

	allocateArray((void **) &m_dCellStart, m_numGridCells * sizeof(uint));
	allocateArray((void **) &m_dCellEnd, m_numGridCells * sizeof(uint));

	sdkCreateTimer(&m_timer);
	setParameters(&m_params);

	m_bInitialized = true;

}

void McaSystem::_finalize() {
	assert(m_bInitialized);

	delete[] m_hPos;
	delete[] m_hPos_prev;
	delete[] m_hAngle;
	delete[] m_hAngle_prev;
	delete[] m_hAccel;
	delete[] m_hAngleAccel;
	delete[] m_hDeform;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;
	delete[] m_hGridParticleHash;
	delete[] m_hGridParticleIndex;
	delete[] m_hExternalParticlesIndices;
	delete[] m_hExternalForceNorms;
	delete[] m_hSelectedParticleIndices;
	delete[] m_hNeighbors;
	delete[] m_hNumNeighb;
	delete[] m_hStress;

	freeArray(m_dPos_prev);
	freeArray(m_dAccel);
	freeArray(m_dVel);
	freeArray(m_dDeform);
	freeArray(m_dAngle);
	freeArray(m_dAngle_prev);
	freeArray(m_dAngleAccel);
	freeArray(m_dStress);

	freeArray(m_dNeighbors);
	freeArray(m_dNumNeighb);

	freeArray(m_dExternalParticlesIndices);
	freeArray(m_dExternalForceNorms);

	freeArray(m_dSelectedParticleIndices);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	if (bUseGL) {
		unregisterGLBufferObject(m_cuda_colorlinksvbo_resource);
		unregisterGLBufferObject(m_cuda_colordeformvbo_resource);
		unregisterGLBufferObject(m_cuda_coloraccelvbo_resource);
		unregisterGLBufferObject(m_cuda_posvbo_resource);

		glDeleteBuffers(1, (const GLuint *) &m_posVbo);
		glDeleteBuffers(1, (const GLuint *) &m_colorLinksVbo);
		glDeleteBuffers(1, (const GLuint *) &m_colorDeformVbo);
		glDeleteBuffers(1, (const GLuint *) &m_colorAccelVbo);
	} else {
		freeArray(m_dPos);
	}
}

void McaSystem::setExternalForceVal(double deltaForce) {
	m_params.externalDeltaForce = deltaForce;
}

void McaSystem::writeSelectedArrayToFile(const char* file_name, ParticleArray arrayType) {
	if (!m_numSelectedParticles)
		return;

	void* array = getArray(arrayType);
	uint* arrayUint;
	double4* arrayDouble4;
	double* arrayDouble;

	std::ofstream file(file_name, std::ios_base::out | std::ios_base::app);

	if (arrayType == ParticleArray::NUM_NEIGHBRS || arrayType == ParticleArray::NEIGHBORS) {
		arrayUint = (uint*) array;
	} else if (arrayType == ParticleArray::STRESS) {
		arrayDouble = (double*) array;
		for (int i = 0; i < m_numSelectedParticles; i++) {
			file << arrayDouble[m_hSelectedParticleIndices[i]] << "\n";
		}
	} else {
		arrayDouble4 = (double4*) array;
		for (int i = 0; i < m_numSelectedParticles; i++) {
			/*printf("accel %f, %f, %f\n", arrayDouble4[m_hSelectedParticleIndices[i]].x,
					arrayDouble4[m_hSelectedParticleIndices[i]].y,
					arrayDouble4[m_hSelectedParticleIndices[i]].z);*/
			file << arrayDouble4[m_hSelectedParticleIndices[i]].x << "\t" <<
					arrayDouble4[m_hSelectedParticleIndices[i]].y << "\t" <<
					arrayDouble4[m_hSelectedParticleIndices[i]].z << "\n";

		}
	}

	file.close();
}

void print_double3(const char* str, double* array, int index) {
	printf("%s: %3.6f, %3.6f, %3.6f\n", str, array[index], array[index + 1], array[index + 2]);
}

void McaSystem::updateVelretTestMode(bool isGPU, int numThreads) {
	assert(m_bInitialized);

	if (isGPU) {
		//указатель на массив позиций частиц
		double *dPos;
		//указатели на массивы цветов для отображения разных режимов визуализации
		double *dColorLinks = 0;
		double *dColorDeform = 0;
		double *dColorAccel = 0;

		//проверка на использование OpenGL, то же самое что и проверка на выполенение в консоли
		if (bUseGL) {
			/*в случае с визуализацией, используем технологию прямого доступа к данным в карте
			минуя оперативную память*/
			dPos = (double *) mapGLBufferObject(&m_cuda_posvbo_resource);
			dColorLinks = (double *) mapGLBufferObject(
					&m_cuda_colorlinksvbo_resource);
			dColorDeform = (double *) mapGLBufferObject(
					&m_cuda_colordeformvbo_resource);
			dColorAccel = (double *) mapGLBufferObject(
					&m_cuda_coloraccelvbo_resource);
		} else {
			/*в случае с консольным режимом, достаточно держать данные все время на карте*/
			dPos = m_dPos;
		}

		/*обновление параметров системы(шаг по времени, режим визуализации, внешние силы и т.д.)
		 * в случае, если пользователь меняет их в ходе выполнения по*/
		setParameters(&m_params);

		/*интегрирование уравнения движения методом Верле*/
		integrateVerletSystemTest(dPos, m_dPos_prev, m_dVel, m_dAccel, m_dAngle,
				m_dAngle_prev, m_dAngleAccel, m_numParticles, numThreads);

		/* Следующие три метода - выполнение алгоритма поиска ближайших соседей
		 * Алгоритм Uniform Sorting Grid - USD. Описан в отчете по курсовой работе в 6ом семестре.
		 * взят с материала */

		/* Вычисление hash для тех клеток(пространство разбивается на сетки, то
		 * есть на множество клеток), в которых находятся частицы
		 * m_dGridParticleHash - массив для хранения хэш клетки
		 * в которой находится частица
		 *
		 * m_dGridParticleIndex - массив для хранения индекса той частицы, по которой
		 * ищется хэш клетки, в которой она находится
		*/
		calcHashTest(m_dGridParticleHash, m_dGridParticleIndex, dPos,
				m_numParticles, numThreads);

		/* отсортировка массивов m_dGridParticleIndex и m_dGridParticleHash
		 * как две связанные пары массивов по значениям m_dGridParticleHash
		 * */
		sortArrayByKeyUint(m_dGridParticleHash, m_dGridParticleIndex,
				m_numParticles);

		/*установка значений для массивов m_dCellStart, m_dCellEnd, указывающие
		 * на начало и на конец индексов одинаковых элементов m_dGridParticleHash
		 * (в массиве m_dGridParticleHash существование k одинаковых элементов указывает на то,
		 * что клетка с данным хэшэм содержит в себе k частиц)*/
		reorderDataAndFindCellStartTest(m_dCellStart, m_dCellEnd,
				m_dGridParticleHash, m_dGridParticleIndex, m_numParticles,
				m_numGridCells, numThreads);

		/*обновление связей частиц и установка значения средней деформации
		 * по всем соседям для каждой частицы */
		updateLinksAndDeformTest(m_dGridParticleIndex, m_dDeform, dPos,
				dColorLinks, dColorDeform, m_dCellStart, m_dCellEnd,
				m_dNeighbors, m_dNumNeighb, m_numParticles, numThreads);

		//reset accelerations
		checkCudaErrors(cudaMemset(m_dAccel, 0, 4 * m_numParticles * sizeof(double)));

		/*взаимодействие частиц*/
		collideTest(m_dStress, m_dVel, m_dAccel, dPos, m_dPos_prev, m_dAngle,
				m_dAngle_prev, m_dAngleAccel, m_dDeform, dColorAccel,
				m_dGridParticleIndex, m_dCellStart, m_dCellEnd, m_dNeighbors,
				m_dNumNeighb, m_numParticles, numThreads);

		/*установка внешних сил*/
		externaForceInteractionTest(m_dExternalParticlesIndices,
								m_dExternalForceNorms, m_dAccel, dColorLinks, dColorDeform,
								dColorAccel, m_numExternalParticles, numThreads);

		if (bUseGL) {
			/*требование технологии прямого доступа к данным при визуализации -
			 *отвязка от "ресурсов" видеокарты перед визуализацией*/
			unmapGLBufferObject(m_cuda_posvbo_resource);
			unmapGLBufferObject(m_cuda_colorlinksvbo_resource);
			unmapGLBufferObject(m_cuda_colordeformvbo_resource);
			unmapGLBufferObject(m_cuda_coloraccelvbo_resource);
		}

	} else {
		double *hPos;
		double *hColorLinks = 0;
		double *hColorDeform = 0;
		double *hColorAccel = 0;

		if (bUseGL) {
			hPos = mapVBO(m_posVbo);
			hColorLinks = mapVBO(m_colorLinksVbo);
			hColorDeform = mapVBO(m_colorDeformVbo);
			hColorAccel = mapVBO(m_colorAccelVbo);
		} else {
			hPos = m_hPos;
		}

		omp_set_num_threads(numThreads);

		// update constants
		setParametersCPU(&m_params);
		// integrate
		integrateVerletSystemCPU(hPos, m_hPos_prev, m_hVel, m_hAccel, m_hAngle,
				m_hAngle_prev, m_hAngleAccel, m_numParticles);

		calcHashCPU(m_hGridParticleHash, m_hGridParticleIndex, hPos,
				m_numParticles);

		setArray(GRID_HASH, (void*)m_hGridParticleHash, 0, m_numParticles);
		setArray(GRID_INDEX, (void*)m_hGridParticleIndex, 0, m_numParticles);
		sortArrayByKeyUint(m_dGridParticleHash, m_dGridParticleIndex,
				m_numParticles);
		getArray(GRID_HASH);
		getArray(GRID_INDEX);

		reorderDataAndFindCellStartCPU(m_hCellStart, m_hCellEnd,
				m_hGridParticleHash, m_hGridParticleIndex, m_numParticles,
				m_numGridCells);

		updateLinksAndDeformCPU(m_hGridParticleIndex, m_hDeform, hPos,
				hColorLinks, hColorDeform, m_hCellStart, m_hCellEnd,
				m_hNeighbors, m_hNumNeighb, m_numParticles);

		//memset(m_hAccel, 0, m_numParticles * 4 * sizeof(double));



		collideCPU(m_hVel, m_hAccel, hPos, m_hPos_prev, m_hAngle,
				m_hAngle_prev, m_hAngleAccel, m_hDeform, hColorAccel,
				m_hGridParticleIndex, m_hCellStart, m_hCellEnd, m_hNeighbors,
				m_hNumNeighb, m_numParticles);
		externalForceInteractionCPU(m_hExternalParticlesIndices,
								m_hExternalForceNorms, m_hAccel, hColorLinks, hColorDeform,
								hColorAccel, m_numExternalParticles);

		if (bUseGL) {
			unmapVBO(m_posVbo);
			unmapVBO(m_colorLinksVbo);
			unmapVBO(m_colorDeformVbo);
			unmapVBO(m_colorAccelVbo);
		}

	}
}


// step the simulation by Verlet method
void McaSystem::updateVerlet() {
	assert(m_bInitialized);

	double *dPos;
	double *dColorLinks = 0;
	double *dColorDeform = 0;
	double *dColorAccel = 0;

	if (bUseGL) {
		dPos = (double *) mapGLBufferObject(&m_cuda_posvbo_resource);
		dColorLinks = (double *) mapGLBufferObject(
				&m_cuda_colorlinksvbo_resource);
		dColorDeform = (double *) mapGLBufferObject(
				&m_cuda_colordeformvbo_resource);
		dColorAccel = (double *) mapGLBufferObject(
				&m_cuda_coloraccelvbo_resource);
	} else {
		dPos = m_dPos;
	}

	// update constants
	setParameters(&m_params);

	// integrate
	integrateVerletSystem(dPos, m_dPos_prev, m_dVel, m_dAccel, m_dAngle,
			m_dAngle_prev, m_dAngleAccel, m_numParticles);

	// calculate grid hash
	calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);

	// sort particles based on hash
	sortArrayByKeyUint(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd,
			m_dGridParticleHash, m_dGridParticleIndex,
			m_numParticles, m_numGridCells);

	//update links
	updateLinksAndDeform(m_dGridParticleIndex, m_dDeform, dPos, dColorLinks, dColorDeform, m_dCellStart, m_dCellEnd, m_dNeighbors, m_dNumNeighb, m_numParticles);

	//reset accelerations
	checkCudaErrors(cudaMemset(m_dAccel, 0, 4 * m_numParticles * sizeof(double)));

	// process collisions
	collide(m_dStress, m_dVel, m_dAccel, dPos, m_dPos_prev, m_dAngle, m_dAngle_prev, m_dAngleAccel, m_dDeform, dColorAccel, m_dGridParticleIndex,
			m_dCellStart, m_dCellEnd, m_dNeighbors, m_dNumNeighb, m_numParticles);

	externaForceInteraction(m_dExternalParticlesIndices, m_dExternalForceNorms, m_dAccel, dColorLinks, dColorDeform, dColorAccel, m_numExternalParticles);

	if (bUseGL) {
		updateSelectedParticles(m_dSelectedParticleIndices, dColorLinks,
				dColorDeform, dColorAccel, m_numSelectedParticles);

		unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_colorlinksvbo_resource);
		unmapGLBufferObject(m_cuda_colordeformvbo_resource);
		unmapGLBufferObject(m_cuda_coloraccelvbo_resource);
	}
}

void McaSystem::dumpGrid() {
	// dump grid information
	copyArrayFromDevice(m_hCellStart, m_dCellStart, 0,
			sizeof(uint) * m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0,
			sizeof(uint) * m_numGridCells);
	uint maxCellSize = 0;

	for (uint i = 0; i < m_numGridCells; i++) {
		if (m_hCellStart[i] != 0xffffffff) {
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize) {
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void McaSystem::dumpParticles(uint start, uint count) {
	// debug
	if (bUseGL) {
		copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource,
					sizeof(double) * 4 * m_numParticles);
	} else {
		copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(double) * 4 * m_numParticles);
	}
	copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(double) * 4 * m_numParticles);

	for (uint i = start; i < start + count; i++) {
		//        printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i * 4 + 0],
				m_hPos[i * 4 + 1], m_hPos[i * 4 + 2], m_hPos[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i * 4 + 0],
				m_hVel[i * 4 + 1], m_hVel[i * 4 + 2], m_hVel[i * 4 + 3]);
	}
}

void *
McaSystem::getArray(ParticleArray array) {
	assert(m_bInitialized);

	void *hdata = 0;
	void *ddata = 0;
	struct cudaGraphicsResource **cuda_vbo_resource = 0;
	int size;
	switch (array) {
	default:
		return 0;
	case POSITION:
		ddata = 0;
		hdata = m_hPos;
		if (bUseGL) {
			cuda_vbo_resource = &m_cuda_posvbo_resource;
		} else {
			ddata = m_dPos;
		}
		size = m_numParticles * 4 * sizeof(double);
		break;
	case ANGLE:
		ddata = m_dAngle;
		hdata = m_hAngle;
		size = m_numParticles * 4 * sizeof(double);
		break;
	case ACCEL:
		ddata = m_dAccel;
		hdata = m_hAccel;
		size = m_numParticles * 4 * sizeof(double);
		break;
	case VELOCITY:
		ddata = m_dVel;
		hdata = m_hVel;
		size = m_numParticles * 4 * sizeof(double);
		break;
	case EXTERNAL_PARTICLES:
		if (m_hExternalParticlesIndices && m_numExternalParticles)
			return m_hExternalParticlesIndices;
		ddata = (double*) m_dExternalParticlesIndices;
		hdata = m_hExternalParticlesIndices;
		size = m_numParticles * sizeof(uint);
		break;
	case EXTERNAL_FORCE_NORM:
		ddata = (double*) m_dExternalForceNorms;
		hdata = m_hExternalForceNorms;
		size = m_numParticles * 4 * sizeof(double);
		break;
	case DEFORM:
		ddata = m_dDeform;
		hdata = m_hDeform;
		size = m_numParticles * sizeof(double);
		break;
	case STRESS:
		ddata = m_dStress;
		hdata = m_hStress;
		size = m_numParticles * sizeof(double);
		break;
	case SELECTED_PARTICLES_INDICES:
		ddata = m_dSelectedParticleIndices;
		hdata = m_hSelectedParticleIndices;
		size = m_numParticles * sizeof(uint);
		break;
	case GRID_HASH:
		ddata = m_dGridParticleHash;
		hdata = m_hGridParticleHash;
		size = m_numParticles * sizeof(uint);
		break;
	case GRID_INDEX:
		ddata = m_dGridParticleIndex;
		hdata = m_hGridParticleIndex;
		size = m_numParticles * sizeof(uint);
		break;
	case NEIGHBORS:
		ddata = m_dNeighbors;
		hdata = m_hNeighbors;
		size = m_numParticles * m_params.mcaMaxNumNeighbors * sizeof(uint);
		break;
	case NUM_NEIGHBRS:
		ddata = m_dNumNeighb;
		hdata = m_hNumNeighb;
		size = m_numParticles * sizeof(uint);
		break;
	}

	copyArrayFromDevice(hdata, ddata, cuda_vbo_resource, size);
	return hdata;
}

void McaSystem::setArray(ParticleArray array, const void *data, int start,
		int count) {
	assert(m_bInitialized);

	switch (array) {
	default:
	case POSITION: {
		if (bUseGL) {
			unregisterGLBufferObject(m_cuda_posvbo_resource);
			glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
			glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(double),
					count * 4 * sizeof(double), data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		} else {
			copyArrayToDevice(m_dPos, data, 0,
							m_numParticles * 4 * sizeof(double));
		}
	}
		break;

	case PRE_POSITION:
		copyArrayToDevice(m_dPos_prev, data, 0,
				m_numParticles * 4 * sizeof(double));
		break;
	case ANGLE:
		copyArrayToDevice(m_dAngle, data, 0,
				m_numParticles * 4 * sizeof(double));
		break;
	case ACCEL:
		copyArrayToDevice(m_dAccel, data, start * 4 * sizeof(double),
				count * 4 * sizeof(double));

		break;
	case EXTERNAL_PARTICLES:
		copyArrayToDevice(m_dExternalParticlesIndices, data,
				start * sizeof(uint), count * sizeof(uint));
		break;
	case EXTERNAL_FORCE_NORM:
		copyArrayToDevice(m_dExternalForceNorms, data,
				start * 4 * sizeof(double), count * 4 * sizeof(double));
		break;
	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(double),
				count * 4 * sizeof(double));
		break;
	case SELECTED_PARTICLES_INDICES:
		copyArrayToDevice(m_dSelectedParticleIndices, data, start  * sizeof(uint),
				count  * sizeof(uint));
		break;
	case GRID_INDEX:
		copyArrayToDevice(m_dGridParticleIndex, data,
				start * sizeof(uint), count * sizeof(uint));
		break;
	case GRID_HASH:
		copyArrayToDevice(m_dGridParticleHash, data,
				start * sizeof(uint), count * sizeof(uint));
		break;
	}
}


inline double frand() {
	return rand() / (double) RAND_MAX;
}

void McaSystem::initHCP(uint numColumn, uint numRow) {
	uint numDepth = m_numParticles / (numColumn * numRow);
	if (m_numParticles % (numColumn * numRow) != 0)
		numDepth++;

	double width = m_params.particleRadius * 2.0;
	double height = 2.0 / sqrt(3.0) * width;
	double spacing = height;

	double radius = width/2.0;

	for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
		for (uint countRow = 0; countRow < numRow; countRow++) {
			for (uint countCol = 0; countCol < numColumn; countCol++) {
				uint index = (countDepth * numColumn * numRow)
						+ (countRow * numColumn) + countCol;

				double x = (2 * countCol + (countRow + countDepth) % 2) * radius
										+ m_params.worldOrigin.x;

				double y = sqrt(3.0) * (countRow + 1.0 / 3.0 * (countDepth % 2)) * radius
						+ m_params.worldOrigin.y;

				double z = 2 * sqrt(6.0) / 3.0 * countDepth * radius
						+ m_params.worldOrigin.z;

				m_hPos[index * 4] = x;
				m_hPos[index * 4 + 1] = y;
				m_hPos[index * 4 + 2] = z;
				m_hPos[index * 4 + 3] = 1.0; //mass

				m_hPos_prev[index * 4] = x;
				m_hPos_prev[index * 4 + 1] = y;
				m_hPos_prev[index * 4 + 2] = 0;
				m_hPos_prev[index * 4 + 3] = 1.0; //mass

				m_hAccel[index * 4] = 0.0;
				m_hAccel[index * 4 + 1] = 0.0;
				m_hAccel[index * 4 + 2] = 0.0;
				m_hAccel[index * 4 + 3] = 0.0;

				m_hAngle[index * 4] = 0.0;
				m_hAngle[index * 4 + 1] = 0.0;
				m_hAngle[index * 4 + 2] = 0.0;
				m_hAngle[index * 4 + 3] = 0.0;
			}
		}
	}

}

void McaSystem::initSquare(uint numColumn) {
	uint numRow = m_numParticles / numColumn;
	if (m_numParticles % numColumn != 0)
		numRow++;

	double width = m_params.particleRadius * 2.0;
	double height = m_params.particleRadius * 2.0;
	double spacing = height;

	/*double offset_x_for_center = (m_params.spaceSize.x - numColumn * spacing)
			/ 2.0;*/

	double offset_x_for_center = 0;

	for (uint countRow = 0; countRow < numRow; countRow++) {
		for (uint countCol = 0; countCol < numColumn; countCol++) {
			uint index = (countRow * numColumn) + countCol;

			double y = height / 2.0 + ((countRow) * (height))
					+ m_params.worldOrigin.y;
			double x = width / 2.0 + ((countCol) * (width))
					+ m_params.worldOrigin.x + offset_x_for_center;

			m_hPos[index * 4] = x;
			m_hPos[index * 4 + 1] = y;
			m_hPos[index * 4 + 2] = 0;
			m_hPos[index * 4 + 3] = 1.0; //mass

			m_hPos_prev[index * 4] = x;
			m_hPos_prev[index * 4 + 1] = y;
			m_hPos_prev[index * 4 + 2] = 0;
			m_hPos_prev[index * 4 + 3] = 1.0; //mass

			m_hAccel[index * 4] = 0.0;
			m_hAccel[index * 4 + 1] = 0.0;
			m_hAccel[index * 4 + 2] = 0.0;
			m_hAccel[index * 4 + 3] = 0.0;

			m_hAngle[index * 4] = 0.0;
			m_hAngle[index * 4 + 1] = 0.0;
			m_hAngle[index * 4 + 2] = 0.0;
			m_hAngle[index * 4 + 3] = 0.0;
		}
	}
}


void McaSystem::initGrid(uint numColumn) {
	uint numRow = m_numParticles / numColumn;
	if (m_numParticles % numColumn != 0)
		numRow++;

	double width = m_params.particleRadius * 2.0;
	double height = 2.0 / sqrt(3.0) * width;
	double spacing = height;

	double offset_x_for_center = (m_params.spaceSize.x - numColumn * spacing)
			/ 2.0;
	double offset_y_for_center = (m_params.spaceSize.y - numRow * spacing)
				/ 2.0;
	//double offset_x_for_center = 0;

	for (uint countRow = 0; countRow < numRow; countRow++) {
		for (uint countCol = 0; countCol < numColumn; countCol++) {
			uint index = (countRow * numColumn) + countCol;

			double y = height / 2.0 + ((countRow) * (height * 3.0 / 4.0))
					+ m_params.worldOrigin.y + offset_y_for_center;
			double x = width / 2.0 + ((countCol) * (width))
					+ m_params.worldOrigin.x + offset_x_for_center;

			if (countRow % 2 != 0)
				x += width / 2.0;
			m_hPos[index * 4] = x;
			m_hPos[index * 4 + 1] = y;
			m_hPos[index * 4 + 2] = 0;
			m_hPos[index * 4 + 3] = 1.0; //mass

			m_hPos_prev[index * 4] = x;
			m_hPos_prev[index * 4 + 1] = y;
			m_hPos_prev[index * 4 + 2] = 0;
			m_hPos_prev[index * 4 + 3] = 1.0; //mass

			m_hAccel[index * 4] = 0.0;
			m_hAccel[index * 4 + 1] = 0.0;
			m_hAccel[index * 4 + 2] = 0.0;
			m_hAccel[index * 4 + 3] = 0.0;

			m_hAngle[index * 4] = 0.0;
			m_hAngle[index * 4 + 1] = 0.0;
			m_hAngle[index * 4 + 2] = 0.0;
			m_hAngle[index * 4 + 3] = 0.0;
		}
	}
}

void McaSystem::resetData() {
	memset(m_hPos, 0, m_numParticles * 4 * sizeof(double));
	memset(m_hPos_prev, 0, m_numParticles * 4 * sizeof(double));
	memset(m_hAccel, 0, m_numParticles * 4 * sizeof(double));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(double));
	memset(m_hDeform, 0, m_numParticles * sizeof(double));
	memset(m_hExternalParticlesIndices, 0, m_numParticles * sizeof(uint));
	memset(m_hExternalForceNorms, 0, m_numParticles * 4 * sizeof(double));
	memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));
	memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

	checkCudaErrors(cudaMemset(m_dVel, 0, sizeof(double) * 4 * m_numParticles));
	checkCudaErrors(cudaMemset(m_dAngle, 0, sizeof(double) * 4 * m_numParticles));
	checkCudaErrors(cudaMemset(m_dAngle_prev, 0, sizeof(double) * 4 * m_numParticles));
	checkCudaErrors(cudaMemset(m_dAngleAccel, 0, sizeof(double) * 4 * m_numParticles));
}

void McaSystem::setExternalForceMode(uint numColumn, uint numRow, ParticleExternalForce mode) {
	if (m_params.mcaD == 2) {
		setExternalForceMode2D(numColumn, mode);
	}
	if (m_params.mcaD == 3) {
		setExternalForceMode3D(numColumn, numRow, mode);
	}
}

void McaSystem::configSystem(uint numColumn, uint numRow,
		ParticleConfig config) {

	resetData();
	switch (config) {
	default:
	case CONFIG_RANDOM: {
		m_params.mcaD = 2;
		m_params.mcaMaxNumNeighbors = 6;
		m_params.automateV = m_params.automateD * m_params.automateD * m_params.automateD * sqrt(3.0) / 2.0;
		m_params.automateS = m_params.automateD * m_params.automateD / sqrt(3.0);

		int p = 0, v = 0, z = 0;
		for (uint i = 0; i < m_numParticles; i++) {
			double point[3];
			point[0] = m_params.spaceSize.x * frand();
			point[1] = m_params.spaceSize.y * frand();
			point[2] = 0;
			m_hPos[p++] = point[0];
			m_hPos[p++] = point[1];
			m_hPos[p++] = 0;
			m_hPos[p++] = 1.0; // mass
			m_hAccel[v++] = 0.0;
			m_hAccel[v++] = 0.0;
			m_hAccel[v++] = 0.0;
			m_hAccel[v++] = 0.0;
			m_hAngle[z++] = 0.0;
			m_hAngle[z++] = 0.0;
			m_hAngle[z++] = 0.0;
			m_hAngle[z++] = 0.0;
		}
	}
		break;
	case CONFIG_SQUARE: {
		m_params.mcaD = 2;
		m_params.mcaMaxNumNeighbors = 4;
		m_params.automateV = m_params.automateD * m_params.automateD
				* m_params.automateD;
		m_params.automateS = m_params.automateD * m_params.automateD;
		initSquare(numColumn);
	}
		break;
	case CONFIG_GRID: {
		m_params.mcaD = 2;
		m_params.mcaMaxNumNeighbors = 6;
		m_params.automateV = m_params.automateD * m_params.automateD * m_params.automateD * sqrt(3.0) / 2.0;
		m_params.automateS = m_params.automateD * m_params.automateD / sqrt(3.0);
		initGrid(numColumn);
	}
		break;
	case CONFIG_HCP: {
		m_params.mcaD = 3;
		m_params.mcaMaxNumNeighbors = 12;
		m_params.automateV = m_params.automateD * m_params.automateD * m_params.automateD / sqrt(2.0);
		m_params.automateS = m_params.automateD * m_params.automateD / (2 * sqrt(2.0));
		initHCP(numColumn, numRow);
	}
		break;
	}

	//m_params.mcaMaxNumNeighbors is changed
	freeArray(m_dNeighbors);
	allocateArray((void **) &m_dNeighbors,
			m_numParticles * m_params.mcaMaxNumNeighbors * sizeof(uint));
	delete[] m_hNeighbors;
	m_hNeighbors = new uint[m_params.mcaMaxNumNeighbors * m_numParticles];

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(PRE_POSITION, m_hPos, 0, m_numParticles);
	setArray(ACCEL, m_hAccel, 0, m_numParticles);
	setArray(ANGLE, m_hAngle, 0, m_numParticles);
}

void McaSystem::configSystem(uint numColumn, uint numRow, ParticleConfig config, ParticleExternalForce mode) {
	configSystem(numColumn, numRow, config);
	setExternalForceMode(numColumn, numRow, mode);

	setParameters(&m_params);
	double* pos;
	if (bUseGL) {
		pos = (double *) mapGLBufferObject(&m_cuda_posvbo_resource);
	} else
		pos = m_dPos;


	initLinks(m_dGridParticleHash, m_dGridParticleIndex, m_numGridCells, pos,
			m_dCellStart, m_dCellEnd, m_dNeighbors, m_dNumNeighb,
			m_numParticles);

	if (bUseGL) {
		unmapGLBufferObject(m_cuda_posvbo_resource);
	}

	m_hNeighbors = (uint*) getArray(NEIGHBORS);
	m_hNumNeighb = (uint*) getArray(NUM_NEIGHBRS);

}

void McaSystem::setExternalForceMode2D(uint numColumn,
		ParticleExternalForce mode) {
	int v = 0;

	uint numRow = m_numParticles / numColumn;
	int h = 0;

	switch (mode) {
	default:
	case EXTERNAL_FORCE_DEFAULT:
		m_numExternalParticles = 0;
		//floor
		for (int i = 0; i < numColumn; i++) {
			m_hExternalParticlesIndices[m_numExternalParticles + i] = i;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}
		m_numExternalParticles += numColumn;

		m_externalForceMode = EXTERNAL_FORCE_DEFAULT;
		break;
	case EXTERNAL_FORCE_CORNER:
		m_numExternalParticles = 1;
		m_hExternalParticlesIndices[0] = m_numParticles - 1;
		m_hExternalForceNorms[v++] = 0.0;
		m_hExternalForceNorms[v++] = -1.0;
		m_hExternalForceNorms[v++] = 0.0;

		for (int i = 0; i < numColumn; i++) {
			m_hExternalParticlesIndices[m_numExternalParticles + i] = i;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}
		m_numExternalParticles += numColumn;

		m_externalForceMode = EXTERNAL_FORCE_CORNER;
		break;
	case EXTERNAL_FORCE_TWO_CORNERS:
		m_numExternalParticles = 2;
		m_hExternalParticlesIndices[0] = m_numParticles - 1;
		m_hExternalParticlesIndices[1] = m_numParticles - numColumn;

		for (int j = 0; j < m_numExternalParticles; j++) {
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = -1.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}

		for (int i = 0; i < numColumn; i++) {
			m_hExternalParticlesIndices[m_numExternalParticles + i] = i;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}
		m_numExternalParticles += numColumn;

		m_externalForceMode = EXTERNAL_FORCE_TWO_CORNERS;
		break;
	case EXTERNAL_FORCE_FORCER:
		m_numExternalParticles = numColumn;
		for (int i = m_numParticles - 1, k = 0; i >= m_numParticles - numColumn;
				i--, k++) {
			m_hExternalParticlesIndices[k] = i;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = -1.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}

		for (int i = 0; i < numColumn; i++) {
			m_hExternalParticlesIndices[m_numExternalParticles + i] = i;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}
		m_numExternalParticles += numColumn;

		m_externalForceMode = EXTERNAL_FORCE_FORCER;
		break;
	case EXTERNAL_FORCE_PUSH_APART:
		m_numExternalParticles = 0;

		if (m_numParticles % numColumn != 0) {
			numRow++;
			h++;
		}

		//left side
		for (int i = 0; i < numRow; i++) {
			m_hExternalParticlesIndices[i] = i * numColumn;
			m_hExternalForceNorms[v++] = -1.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
			m_numExternalParticles++;
		}

		//right side
		for (int i = 0; i < numRow - h; i++) {
			m_hExternalParticlesIndices[i + numRow] = (i + 1) * numColumn - 1;
			m_hExternalForceNorms[v++] = 1.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
			m_numExternalParticles++;
		}

		m_externalForceMode = EXTERNAL_FORCE_PUSH_APART;
		break;
	case EXTERNAL_FORCE_PULL_APART:
		m_numExternalParticles = 0;
		if (m_numParticles % numColumn != 0) {
			numRow++;
			h++;
		}

		//left side
		for (int i = 0; i < numRow; i++) {
			m_hExternalParticlesIndices[i] = i * numColumn;
			m_hExternalForceNorms[v++] = 1.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
			m_numExternalParticles++;
		}

		//right side
		for (int i = 0; i < numRow - h; i++) {
			m_hExternalParticlesIndices[i + numRow] = (i + 1) * numColumn - 1;
			m_hExternalForceNorms[v++] = -1.0;
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
			m_numExternalParticles++;
		}

		m_externalForceMode = EXTERNAL_FORCE_PUSH_APART;
		break;
	}

	setArray(EXTERNAL_PARTICLES, m_hExternalParticlesIndices, 0, m_numParticles);
	setArray(EXTERNAL_FORCE_NORM, m_hExternalForceNorms, 0, m_numParticles);
}


void McaSystem::setExternalForceMode3D(uint numColumn, uint numRow, ParticleExternalForce mode) {
	int v = 0;

	uint numDepth = m_numParticles / (numColumn * numRow);
	int h = 0;

	switch (mode) {
	default:
	case EXTERNAL_FORCE_DEFAULT:
		m_numExternalParticles = 0;
		//floor
		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					if (countRow != 0)
						continue;
					m_hExternalParticlesIndices[m_numExternalParticles] = index;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					v++;
					m_numExternalParticles++;
				}
			}
		}

		m_externalForceMode = EXTERNAL_FORCE_DEFAULT;
		break;
	case EXTERNAL_FORCE_CORNER:
		m_numExternalParticles = 1;
		m_hExternalParticlesIndices[0] = m_numParticles - 1;
		m_hExternalForceNorms[v++] = 0.0;
		m_hExternalForceNorms[v++] = -1.0;
		m_hExternalForceNorms[v++] = 0.0;
		v++;

		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					if (countRow != 0)
						continue;
					m_hExternalParticlesIndices[m_numExternalParticles] = index;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					v++;
					m_numExternalParticles++;
				}
			}
		}

		m_externalForceMode = EXTERNAL_FORCE_CORNER;
		break;
	case EXTERNAL_FORCE_TWO_CORNERS:
		m_numExternalParticles = 2;
		m_hExternalParticlesIndices[0] = m_numParticles - 1;
		m_hExternalParticlesIndices[1] = m_numParticles - numColumn;

		for (int j = 0; j < m_numExternalParticles; j++) {
			m_hExternalForceNorms[v++] = 0.0;
			m_hExternalForceNorms[v++] = -1.0;
			m_hExternalForceNorms[v++] = 0.0;
			v++;
		}

		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					if (countRow != 0)
						continue;
					m_hExternalParticlesIndices[m_numExternalParticles] = index;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					m_hExternalForceNorms[v++] = 0.0;
					v++;
					m_numExternalParticles++;
				}
			}
		}

		m_externalForceMode = EXTERNAL_FORCE_TWO_CORNERS;
		break;
	case EXTERNAL_FORCE_FORCER:
		m_numExternalParticles = 0;
		v = 0;
		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					//floor
					if (countRow == 0) {
						m_hExternalParticlesIndices[m_numExternalParticles++] = index;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;

						continue;
					}
					//forcer
					if (countRow == numRow - 1) {
						m_hExternalParticlesIndices[m_numExternalParticles++] = index;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = -1.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;
						//printf("HOST: partIndex = %d\n", index);

					}
				}
			}
		}

		m_externalForceMode = EXTERNAL_FORCE_FORCER;
		break;
	case EXTERNAL_FORCE_PUSH_APART: {
		m_numExternalParticles = 0;

		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					//left side
					if (countCol == 0) {
						m_hExternalParticlesIndices[m_numExternalParticles] = index;
						m_hExternalForceNorms[v++] = -1.0;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;
						m_numExternalParticles++;
						continue;
					}
					//right side
					if (countCol == numColumn - 1) {
						m_hExternalParticlesIndices[m_numExternalParticles] = index;
						m_hExternalForceNorms[v++] = 1.0;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;
						m_numExternalParticles++;
					}
				}
			}
		}
		m_externalForceMode = EXTERNAL_FORCE_PUSH_APART;
	}
		break;
	case EXTERNAL_FORCE_PULL_APART:
		m_numExternalParticles = 0;

		for (uint countDepth = 0; countDepth < numDepth; countDepth++) {
			for (uint countRow = 0; countRow < numRow; countRow++) {
				for (uint countCol = 0; countCol < numColumn; countCol++) {
					uint index = (countDepth * numColumn * numRow)
							+ (countRow * numColumn) + countCol;
					//left side
					if (countCol == 0) {
						m_hExternalParticlesIndices[m_numExternalParticles] = index;
						m_hExternalForceNorms[v++] = 1.0;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;
						m_numExternalParticles++;
						continue;
					}
					//right side
					if (countCol == numColumn - 1) {
						m_hExternalParticlesIndices[m_numExternalParticles] = index;
						m_hExternalForceNorms[v++] = -1.0;
						m_hExternalForceNorms[v++] = 0.0;
						m_hExternalForceNorms[v++] = 0.0;
						v++;
						m_numExternalParticles++;
					}
				}
			}
		}
		m_externalForceMode = EXTERNAL_FORCE_PUSH_APART;
		break;
	}
	setArray(EXTERNAL_FORCE_NORM, m_hExternalForceNorms, 0, m_numParticles);
	setArray(EXTERNAL_PARTICLES, m_hExternalParticlesIndices, 0, m_numParticles);
}

void McaSystem::selectParticle(double3 rayOrigin, double3 rayEnd) {
	assert(m_bInitialized);

	double *dPos;
	double *dColorLinks;
	double *dColorDeform;
	double *dColorAccel;

	bool result = false;

	dPos = (double *) mapGLBufferObject(&m_cuda_posvbo_resource);
	dColorLinks = (double *) mapGLBufferObject(&m_cuda_colorlinksvbo_resource);
	dColorDeform = (double *) mapGLBufferObject(
			&m_cuda_colordeformvbo_resource);
	dColorAccel = (double *) mapGLBufferObject(&m_cuda_coloraccelvbo_resource);

	uint selectedParticleIndex = -1;

	result = selectParticleIndex(rayOrigin, rayEnd, dPos, m_numParticles, selectedParticleIndex);

	if (result) {
		bool isSelectedAlready = false;

		//remove previously selected particle
		for (int i = 0; i < m_numSelectedParticles; i++) {
			if (m_hSelectedParticleIndices[i] == selectedParticleIndex) {
				m_hSelectedParticleIndices[i] = m_hSelectedParticleIndices[--m_numSelectedParticles];
				isSelectedAlready = true;
				break;
			}
		}
		//is novice selected particle
		if (!isSelectedAlready) {
			//is m_hSelectedParticleIndices full
			if (m_numSelectedParticles == m_capacitySelectedParticlesArray) {
				m_capacitySelectedParticlesArray = 2 * m_capacitySelectedParticlesArray;
				uint* m_newSelectedArray = new uint[m_capacitySelectedParticlesArray];
				for (int i = 0; i < m_numSelectedParticles; i++) {
					m_newSelectedArray[i] = m_hSelectedParticleIndices[i];
				}
				delete m_hSelectedParticleIndices;
				m_hSelectedParticleIndices = m_newSelectedArray;
			}
			m_hSelectedParticleIndices[m_numSelectedParticles++] = selectedParticleIndex;
			freeArray(m_dSelectedParticleIndices);
			allocateArray((void **) &m_dSelectedParticleIndices, m_capacitySelectedParticlesArray * sizeof(uint));
		}

		setArray(SELECTED_PARTICLES_INDICES, m_hSelectedParticleIndices, 0, m_numSelectedParticles);
	}

	unmapGLBufferObject(m_cuda_coloraccelvbo_resource);
	unmapGLBufferObject(m_cuda_colordeformvbo_resource);
	unmapGLBufferObject(m_cuda_colorlinksvbo_resource);
	unmapGLBufferObject(m_cuda_posvbo_resource);
}


