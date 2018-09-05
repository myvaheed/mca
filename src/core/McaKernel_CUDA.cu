// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

/*#include <cuda_runtime.h>*/
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "McaKernel_impl_CUDA.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

	void setParameters(McaParams *hostParams)
    {
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(McaParams)));
    }

	void copyArrayToDeviceFromDevice(void *dst, const void *src,
			struct cudaGraphicsResource **cuda_vbo_resource, int size) {

		if (cuda_vbo_resource) {
			src = mapGLBufferObject(cuda_vbo_resource);
		}

		checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));

		if (cuda_vbo_resource) {
			unmapGLBufferObject(*cuda_vbo_resource);
		}
	}

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
    	if (n == 0) {
    		numThreads = 1;
    		numBlocks = 1;
    		return;
    	}
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

	void integrateVerletSystem(double *pos, double *pos_prev, double *vel, double *accel,
			double *theta,
			double *thetaPrev,
			double *thetaAccel,
			uint numParticles) {
		thrust::device_ptr<double4> d_pos4((double4 *) pos);
		thrust::device_ptr<double4> d_pos_prev4((double4 *) pos_prev);
		thrust::device_ptr<double4> d_vel4((double4 *) vel);
		thrust::device_ptr<double4> d_accel4((double4 *) accel);

		thrust::device_ptr<double4> d_theta_4((double4 *) theta);
		thrust::device_ptr<double4> d_theta_prev4((double4 *) thetaPrev);
		thrust::device_ptr<double4> d_theta_accel4((double4 *) thetaAccel);


		thrust::for_each(
				thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_pos_prev4, d_vel4, d_accel4,
						d_theta_4, d_theta_prev4, d_theta_accel4)),
				thrust::make_zip_iterator(
						thrust::make_tuple(d_pos4 + numParticles, d_pos_prev4 + numParticles,
								d_vel4 + numParticles, d_accel4 + numParticles, d_theta_4 + numParticles,
								d_theta_prev4 + numParticles, d_theta_accel4 + numParticles)),
				integrate_verlet_functor());
	}

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  double *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (double4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

	void sortArrayByKeyUint(uint *dArrayKey, uint *dSortableArray, uint size) {
		thrust::sort_by_key(thrust::device_ptr<uint>(dArrayKey),
				thrust::device_ptr<uint>(dArrayKey + size),
				thrust::device_ptr<uint>(dSortableArray));
	}

	void sortArrayByKeyDouble(double *dArrayKey, uint *dSortableArray,
			uint size) {
		thrust::sort_by_key(thrust::device_ptr<double>(dArrayKey),
				thrust::device_ptr<double>(dArrayKey + size),
				thrust::device_ptr<uint>(dSortableArray));
	}

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);
        //printf("cellStart = %d\n", cellStart);
        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            gridParticleHash,
            gridParticleIndex,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    }

    void updateLinksAndDeform(
    		uint *gridParticleIndex,
    		double *deformation,
            double *oldPositions,
            double *colorLinks,
            double *colorDeform,
            uint  *cellStart,
            uint  *cellEnd,
            uint* neighbrs, uint* numNeighb,
            uint   numParticles) {
    	uint numThreads, numBlocks;
    	computeGridSize(numParticles, 256, numBlocks, numThreads);

    	//computeGridSize(numParticles * 27, 256, numBlocks, numThreads);

        // set all data to zero
		checkCudaErrors(cudaMemset(deformation, 0, numParticles * sizeof(double)));
		if (colorLinks) {
			checkCudaErrors(
					cudaMemset(colorLinks, 0,
							4 * numParticles * sizeof(double)));
			checkCudaErrors(
					cudaMemset(colorDeform, 0,
							4 * numParticles * sizeof(double)));
		}

		// execute the kernel
		/*updateLinksAndDeformChangedD<<< numBlocks, numThreads >>>(
				gridParticleIndex,
				deformation,
				(double4 *)oldPositions, (double4 *)colorLinks, (double4 *)colorDeform,
				cellStart,
				cellEnd,
				neighbrs, numNeighb,
				numParticles);*/

		// execute the kernel
		updateLinksAndDeformD<<< numBlocks, numThreads >>>(
				gridParticleIndex,
				deformation,
				(double4 *)oldPositions, (double4 *)colorLinks, (double4 *)colorDeform,
				cellStart,
				cellEnd,
				neighbrs, numNeighb,
				numParticles);
		getLastCudaError("Kernel execution failed: updateLinksAndDeformD");
    }

    void initLinks(		uint *gridParticleHash,
            			uint  *gridParticleIndex,
            			uint numCells,
                        double *oldPositions,
                        uint  *cellStart,
                        uint  *cellEnd,
                        uint* neighbrs, uint* numNeighb,
                        uint   numParticles) {

    	//exec UGS algorithm
    	calcHash(gridParticleHash, gridParticleIndex, oldPositions, numParticles);
    	sortArrayByKeyUint(gridParticleHash, gridParticleIndex, numParticles);
    	reorderDataAndFindCellStart(cellStart, cellEnd, gridParticleHash, gridParticleIndex, numParticles, numCells);

    	uint numThreads, numBlocks;

    	//computeGridSize(numParticles, 32, numBlocks, numThreads);
    	computeGridSize(numParticles * 27, 32, numBlocks, numThreads);
    	//computeGridSize(numParticles * 32, 32, numBlocks, numThreads);

    	//reset numNeighb
		checkCudaErrors(cudaMemset(numNeighb, 0, numParticles * sizeof(uint)));

		float ms;
		cudaEvent_t startEvent, stopEvent;
		checkCudaErrors(cudaEventCreate(&startEvent));
		checkCudaErrors(cudaEventCreate(&stopEvent));

		checkCudaErrors( cudaEventRecord(startEvent,0) );

		initLinksChangedD<<< numBlocks, numThreads >>>(
						gridParticleIndex,
						(double4 *)oldPositions,
						cellStart,
						cellEnd,
						neighbrs, numNeighb,
						numParticles);

		// execute the kernel
		/*initLinksD<<< numBlocks, numThreads >>>(
				gridParticleIndex,
				(double4 *)oldPositions,
				cellStart,
				cellEnd,
				neighbrs, numNeighb,
				numParticles);*/

		checkCudaErrors(cudaEventRecord(stopEvent, 0));
		checkCudaErrors(cudaEventSynchronize(stopEvent));
		checkCudaErrors(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("%f ms\n", ms);

		getLastCudaError("Kernel execution failed: initLinks");
    }

    void externaForceInteraction(uint* externalParticleIndex,
    		double* externalForceNorms,
    		double *accelerations,
    		double *colorLinks,
    		double *colorDeform,
    		double *colorAccel,
    		uint numExternalParticles) {
		// thread per particle
		uint numThreads, numBlocks;

		computeGridSize(numExternalParticles, 256, numBlocks, numThreads);

		// execute the kernel
		externalForceInteractionD<<< numBlocks, numThreads >>> (
				externalParticleIndex,
				(double4 *) externalForceNorms,
				(double4 *) accelerations,
				(double4 *) colorLinks,
				(double4 *)	colorDeform,
				(double4 *) colorAccel,
				numExternalParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
    }

    void collide(double *stress,
    			 double *vel,
    			 double *newAccel,
                 double *sortedPos,
                 double *prevPos,
                 double *angles,
                 double *prevAngles,
                 double *angAccelerations,
                 double *deformation,
                 double *colorAccel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint* neighbrs, uint* numNeighb,
                 uint   numParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((double*)stress,
        									  (double4 *)vel,
        									  (double4 *)newAccel,
                                              (double4 *)sortedPos,
                                              (double4 *)prevPos,
                                              (double4 *)angles,
                                              (double4 *)prevAngles,
                                              (double4 *)angAccelerations,
                                              deformation,
                                              (double4 *)colorAccel,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
                                              neighbrs, numNeighb,
                                              numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }



    bool selectParticleIndex(double3 rayBegin, double3 rayEnd, double* positions, uint numParticles, uint& selectedParticleIndex) {

    	uint* dSelectedParticlesIndicesTemp;
    	double* dDistToCamera;
    	uint* dFoundCollision;

    	allocateArray((void **) &dSelectedParticlesIndicesTemp, numParticles * sizeof(uint));
    	allocateArray((void **) &dDistToCamera, numParticles * sizeof(double));
    	allocateArray((void **) &dFoundCollision, sizeof(uint));

    	checkCudaErrors(cudaMemset(dFoundCollision, 0, sizeof(uint)));

    	uint numThreads, numBlocks;
    	computeGridSize(numParticles, 256, numBlocks, numThreads);
    	intersectionParticleD<<< numBlocks, numThreads >>>(
    			rayBegin,
    			rayEnd,
    			dSelectedParticlesIndicesTemp,
    			dDistToCamera,
    			(double4*) positions,
    			numParticles,
    			dFoundCollision);

    	uint* hFoundCollision = new uint;

    	copyArrayFromDevice(hFoundCollision, dFoundCollision, 0, sizeof(uint));
    	if (!(*hFoundCollision)) {
			freeArray(dSelectedParticlesIndicesTemp);
			freeArray(dDistToCamera);
			freeArray(dFoundCollision);
			delete hFoundCollision;
    		return false;
    	}

    	sortArrayByKeyDouble(dDistToCamera, dSelectedParticlesIndicesTemp, numParticles);

    	uint* selectedIndex = new uint;
    	copyArrayFromDevice(selectedIndex, dSelectedParticlesIndicesTemp, 0, sizeof(uint));
    	selectedParticleIndex = *selectedIndex;

    	freeArray(dFoundCollision);
    	freeArray(dSelectedParticlesIndicesTemp);
    	freeArray(dDistToCamera);
    	delete selectedIndex;
    	delete hFoundCollision;

    	// check if kernel invocation generated an error
    	getLastCudaError("Kernel execution failed");

    	return true;
    }

	void updateSelectedParticles(uint* selectedParticlesIndices, double* colorLinks, double* colorDeform, double* colorAccel, uint numSelectedParticles) {
		// thread per particle
		uint numThreads, numBlocks;
		computeGridSize(numSelectedParticles, 256, numBlocks, numThreads);

		// execute the kernel
		updateSelectedParticlesD<<< numBlocks, numThreads >>>(
				selectedParticlesIndices,
				(double4 *) colorLinks,
				(double4 *)	colorDeform,
				(double4 *) colorAccel,
				numSelectedParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	/*FUNCTIONS FOR TESTING*/
	void integrateVerletSystemTest(double *pos, double *pos_prev, double *vel,
			double *accel, double *theta, double *thetaPrev, double *thetaAccel,
			uint numParticles, uint numThreads) {

		//numThreads - общее количество нитей, без учета количества блоков

		//numThreads == 0 ~ numThreads==All threads.
		if (numThreads == 0) {
			/*вызов метода, предназначенная исполнению
			 * при обычном режиме выполнения программы (не тестирование)*/
			integrateVerletSystem(pos, pos_prev, vel, accel, theta, thetaPrev, thetaAccel, numParticles);
			return;
		}

		/*количество блоков и количество нитей в каждом блоке (понятия из CUDA)*/
		uint numBlocks, numDeviceThreads;
		/*определение количества блоков и нитей в каждом блоке исходя от общего количества нитей*/
		/*Устанавливается 256 нитей в каждом блоке, если общее количество нитей >256, иначе
		 *blockSize=1, numDeviceThreads=numThreads*/
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		/*вызов kernel-функции, которая выполняется на карте*/
		integrateVerletFunctionTestD<<<numBlocks, numDeviceThreads>>>(
				(double4*) pos,
				(double4*) pos_prev,
				(double4*) vel,
				(double4*) accel,
				(double4*) theta,
				(double4*) thetaPrev,
				(double4*) thetaAccel,
				numParticles, numThreads);
		getLastCudaError("Kernel execution failed");
	}

	void calcHashTest(uint *gridParticleHash, uint *gridParticleIndex,
			double *pos, int numParticles, uint numThreads) {

		if (numThreads == 0) {
			calcHash(gridParticleHash, gridParticleIndex, pos, numParticles);
			return;
		}

		uint numBlocks, numDeviceThreads;
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		calcHashTestD<<< numBlocks, numDeviceThreads >>>(gridParticleHash,
				gridParticleIndex,
				(double4 *) pos,
				numParticles, numThreads);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void reorderDataAndFindCellStartTest(uint *cellStart, uint *cellEnd,
			uint *gridParticleHash, uint *gridParticleIndex, uint numParticles,
			uint numCells, uint numThreads) {

		if (numThreads == 0) {
			reorderDataAndFindCellStart(cellStart, cellEnd, gridParticleHash,
					gridParticleIndex, numParticles, numCells);
			return;
		}
		// set all cells to empty
		checkCudaErrors(
				cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

		uint numBlocks, numDeviceThreads;
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		uint smemSize = sizeof(uint) * (numDeviceThreads + 1);


		reorderDataAndFindCellStartTestD<<< numBlocks, numDeviceThreads, smemSize>>>(
				cellStart,
				cellEnd,
				gridParticleHash,
				gridParticleIndex,
				numParticles, numThreads);
		getLastCudaError(
				"Kernel execution failed: reorderDataAndFindCellStartD");
	}

	void updateLinksAndDeformTest(uint *gridParticleIndex, double *deformation,
			double *oldPositions, double *colorLinks, double *colorDeform,
			uint *cellStart, uint *cellEnd, uint* neighbrs, uint* numNeighb,
			uint numParticles, uint numThreads) {

		if (numThreads == 0) {
			updateLinksAndDeform(gridParticleIndex, deformation, oldPositions, colorLinks,
					colorDeform, cellStart, cellEnd, neighbrs, numNeighb, numParticles);
			return;
		}

		// set all data to zero
		checkCudaErrors(
				cudaMemset(deformation, 0, numParticles * sizeof(double)));
		if (colorLinks) {
			checkCudaErrors(
					cudaMemset(colorLinks, 0,
							4 * numParticles * sizeof(double)));
			checkCudaErrors(
					cudaMemset(colorDeform, 0,
							4 * numParticles * sizeof(double)));
		}

		uint numBlocks, numDeviceThreads;
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		updateLinksAndDeformTestD<<< numBlocks, numDeviceThreads>>>(
				gridParticleIndex,
				deformation,
				(double4 *)oldPositions, (double4 *)colorLinks, (double4 *)colorDeform,
				cellStart,
				cellEnd,
				neighbrs, numNeighb,
				numParticles, numThreads);
		getLastCudaError(
				"Kernel execution failed: reorderDataAndFindCellStartD");
	}

	void externaForceInteractionTest(uint* externalParticleIndex,
			double* externalForceNorms, double *accelerations,
			double *colorLinks, double *colorDeform, double *colorAccel,
			uint numExternalParticles, uint numThreads) {

		if (numThreads == 0) {
			externaForceInteraction(externalParticleIndex, externalForceNorms, accelerations,
					colorLinks, colorDeform, colorAccel, numExternalParticles);
			return;
		}
		uint numBlocks, numDeviceThreads;
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		externalForceInteractionTestD<<< numBlocks, numDeviceThreads >>> (
				externalParticleIndex,
				(double4 *) externalForceNorms,
				(double4 *) accelerations,
				(double4 *) colorLinks,
				(double4 *) colorDeform,
				(double4 *) colorAccel,
				numExternalParticles, numThreads);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void collideTest(double* stress, double *vel, double *newAccel, double *sortedPos,
			double *prevPos, double *angles, double *prevAngles,
			double *angAccelerations, double *deformation, double *colorAccel,
			uint *gridParticleIndex, uint *cellStart, uint *cellEnd,
			uint* neighbrs, uint* numNeighb, uint numParticles,
			uint numThreads) {

		if (numThreads == 0) {
			//throw new std::exception();
			collide(stress, vel, newAccel, sortedPos, prevPos, angles, prevAngles,
					angAccelerations, deformation, colorAccel, gridParticleIndex,
					cellStart, cellEnd, neighbrs, numNeighb, numParticles);
			return;
		}

		uint numBlocks, numDeviceThreads;
		computeGridSize(numThreads, 256, numBlocks, numDeviceThreads);

		collideTestD<<< numBlocks, numDeviceThreads >>>((double4 *)vel,
				(double4 *)newAccel,
				(double4 *)sortedPos,
				(double4 *)prevPos,
				(double4 *)angles,
				(double4 *)prevAngles,
				(double4 *)angAccelerations,
				deformation,
				(double4 *)colorAccel,
				gridParticleIndex,
				cellStart,
				cellEnd,
				neighbrs, numNeighb,
				numParticles, numThreads);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}
}   // extern "C"
