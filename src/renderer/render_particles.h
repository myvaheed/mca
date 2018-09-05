
#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

#include "vector_types.h"

class ParticleRenderer
{
public:

		enum ColorMode
	        {
	            COLOR_VELOCITY_SPACE,
	            COLOR_DEFORM,
	            COLOR_LINKS,
	            COLOR_TENSION,
	            COLOR_NUM_MODES
	        };

        ParticleRenderer();
        ~ParticleRenderer();

        void setPositions(double *pos, int numParticles);
        void setData(double *pos, double *angles, double *accel, double *vel, uint* numNeighbrs,  uint* externalParticles,int numParticles, int numExternalParticles);
        void setData(unsigned int vboPos, unsigned int vboColorLinks, unsigned int vboColorDeform, unsigned int vboColorAccel, int numParticles);
        void setVertexBuffer(unsigned int vbo, int numParticles);
        void setColorMode(ColorMode colorMode);

        void setColliderPosRad(double3 pos, double rad) {
        	colliderPos = pos;
        	colliderRad = rad;
        }

        void setColorBuffer(unsigned int vbo)
        {
            m_vboColorLinks = vbo;
        }



        void display(ColorMode mode = COLOR_VELOCITY_SPACE);
        void displayGrid();

        bool isNormalizedVelVector() {
        	return normalizeVelSpace;
        }

        void setNormalizedVelVector(bool arg) {
        	normalizeVelSpace = arg;
        }

    	bool getIsTransparancy() {
    		return isTransparency;
    	}

    	void setIsTransparancy(bool arg) {
    		isTransparency = arg;
    	}

        void setPointSize(double size)
        {
            m_pointSize = size;
        }
        void setParticleRadius(double r)
        {
            m_particleRadius = r;
        }
        void setFOV(double fov)
        {
            m_fov = fov;
        }
        void setWindowSize(int w, int h)
        {
            m_window_w = w;
            m_window_h = h;
        }

    protected: // methods
        void _initGL();
        void _drawPoints();
        void _drawAngleLines();
        void _drawVelLines();
        GLuint _compileProgram(const char *vsource, const char *fsource);

    protected: // data

        double *m_pos;
        double *m_angles;
        double *m_vel;
        double *m_accelerations;
        uint *m_numNeighbrs;
        uint *m_externalParticles;
        int m_numParticles;
        int m_numExternalParticles;

        double m_pointSize;
        double m_particleRadius;
        double m_fov;
        int m_window_w, m_window_h;
        bool normalizeVelSpace;
        bool isTransparency;


        double3 colliderPos;
        double colliderRad;

        GLuint m_program;

        ColorMode m_colorMode;
        GLuint m_vboPos;

        GLuint m_vboColor;
        GLuint m_vboColorLinks;
        GLuint m_vboColorDeform;
        GLuint m_vboColorAccel;

};

#endif //__ RENDER_PARTICLES__
