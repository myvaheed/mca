#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

#define MAX_TENSION_COLOR 600
#define MIN_SELECTED_PARTICLES_SIZE 6
#define EPSILON 0.00000001
#define MIN_GPU_NUM_THREADS 16

#define SOFTFVAL 0

// simulation parameters
struct McaParams
{
    double3 colliderPos;
    double  colliderRadius;

    double3 gravity;
    double timeStep;
    double externalDeltaForce;

    double particleRadius;
    double automateMass;
    double automateD; //automate size
    double automateV; //automate volume
    double automateS; //square of contact
    double mcaRmin; //parameter of switching of automate state
    double mcaRmax;
    double mcaIsBrittle;
    double mcaShear;
    double mcaG; // Shear modulus
    double mcaK; // Bulk modulus
    double automateE; // Young's modulus
    double automateMu; // Poisson's ratio
    uint mcaD; // space dimension
    uint mcaMaxNumNeighbors; //max num neighbors
    double bounceCollision; //[0..1] - 0 - absolutely inelastic, 1 - abs.elastic

    uint3 gridSize;
    double3 worldOrigin;
    double3 spaceSize;
    double3 cellSize;
    uint numGridCells;

    uint maxParticlesPerCell;

    double spring;

    double boundaryDamping; //[0.5..1]

};



#endif
