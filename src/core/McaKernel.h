extern "C"
{

    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, size_t size);
    void freeArray(void *devPtr);

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDeviceFromDevice(void *dst, const void *src,
    			struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(McaParams *hostParams);

    void integrateVerletSystem(double *pos, double* pos_prev,
                             double *vel,
                             double *accel,
                             double *theta,
                             double *thetaPrev,
                             double *thetaAccel,
                             uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  double *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     uint   numParticles,
                                     uint   numCells);


    void updateLinksAndDeform(
    			uint  *gridParticleIndex,
    			double *deformation,
                double *oldPositions,
                double *colorLinks,
                double *colorDeform,
                uint  *cellStart,
                uint  *cellEnd,
                uint* neighbrs, uint* numNeighb,
                uint   numParticles);

    void initLinks(		uint *gridParticleHash,
                			uint  *gridParticleIndex,
                			uint numCells,
                            double *oldPositions,
                            uint  *cellStart,
                            uint  *cellEnd,
                            uint* neighbrs, uint* numNeighb,
                            uint   numParticles);

    void externaForceInteraction(uint* externalParticleIndex, double* externalForceNorms, double *accelerations, double* colorLinks, double* colorDeform, double* colorAccel, uint numExternalParticles);


    bool selectParticleIndex(double3 rayBegin, double3 rayEnd, double* positions, uint numParticles, uint& selectedParticleIndex);
    void updateSelectedParticles(uint* selectedParticlesIndices, double* colorLinks, double* colorDeform, double* colorAccel, uint numSelectedParticles);

    void collide(double *stress,
    			 double *velocity,
    			 double *newAccel,
                 double *oldPositions,
                 double *prevPos,
                 double *angles,
                 double *prevAngles,
                 double *angAccelerations,
                 double *deformation,
                 double *colorAccel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint  *neighbrs,
                 uint  *numNeighb,
                 uint   numParticles);


    void sortArrayByKeyUint(uint *dArrayKey, uint *dSortableArray, uint size);
    void sortArrayByKeyDouble(double *dArrayKey, uint *dSortableArray, uint size);


    /*FUNCTIONS FOR TESTING*/
    void integrateVerletSystemTest(double *pos, double* pos_prev,
                                 double *vel,
                                 double *accel,
                                 double *theta,
                                 double *thetaPrev,
                                 double *thetaAccel,
                                 uint numParticles, uint numThreads);
    void calcHashTest(uint  *gridParticleHash,
                      uint  *gridParticleIndex,
                      double *pos,
                      int numParticles, uint numThreads);

    void reorderDataAndFindCellStartTest(uint *cellStart, uint *cellEnd,
		uint *gridParticleHash, uint *gridParticleIndex, uint numParticles,
		uint numCells, uint numThreads);

    void updateLinksAndDeformTest(
        			uint  *gridParticleIndex,
        			double *deformation,
                    double *oldPositions,
                    double *colorLinks,
                    double *colorDeform,
                    uint  *cellStart,
                    uint  *cellEnd,
                    uint* neighbrs, uint* numNeighb,
                    uint   numParticles, uint numThreads);

        void externaForceInteractionTest(uint* externalParticleIndex,
        		double* externalForceNorms,
        		double *accelerations,
        		double* colorLinks,
        		double* colorDeform,
        		double* colorAccel,
        		uint numExternalParticles,
        		uint numThreads);
        void collideTest(double *stress,
        				 double *velocity,
            			 double *newAccel,
                         double *oldPositions,
                         double *prevPos,
                         double *angles,
                         double *prevAngles,
                         double *angAccelerations,
                         double *deformation,
                         double *colorAccel,
                         uint  *gridParticleIndex,
                         uint  *cellStart,
                         uint  *cellEnd,
                         uint  *neighbrs,
                         uint  *numNeighb,
                         uint   numParticles,
                         uint   numThreads);
}
