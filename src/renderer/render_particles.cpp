#include <math.h>
#include <assert.h>
#include <stdio.h>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <GL/freeglut.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif



ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0),
      m_particleRadius(0.125 * 0.5),
      m_program(0),
      m_vboPos(0),
      m_vboColor(0),
	  normalizeVelSpace(true),
	  isTransparency(false)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{

}

void ParticleRenderer::setPositions(double *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setData(unsigned int vboPos, unsigned int vboColorLinks, unsigned int vboColorDeform, unsigned int vboColorAccel, int numParticles) {
	m_vboPos = vboPos;
	m_vboColorLinks = vboColorLinks;
	m_vboColorDeform = vboColorDeform;
	m_vboColorAccel = vboColorAccel;
	m_numParticles = numParticles;
}

void ParticleRenderer::setData(double *pos, double *angles, double *vel, double *accel, uint* numNeighbrs, uint* externalParticles, int numParticles, int numExternalParticles)
{
    m_pos = pos;
    m_angles = angles;
    m_vel = vel;
    m_accelerations = accel;
    m_numNeighbrs = numNeighbrs;
    m_externalParticles = externalParticles;
    m_numParticles = numParticles;
    m_numExternalParticles = numExternalParticles;
}

void ParticleRenderer::setColorMode(ColorMode mode) {
	switch(mode) {
	case COLOR_TENSION:
		m_vboColor = m_vboColorAccel;
		break;
	case COLOR_DEFORM:
		m_vboColor = m_vboColorDeform;
		break;
	case COLOR_LINKS:
		m_vboColor = m_vboColorLinks;
		break;
	}

	m_colorMode = mode;
}

void ParticleRenderer::setVertexBuffer(unsigned int vboPos, int numParticles)
{
    m_vboPos = vboPos;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{

	glBindBuffer(GL_ARRAY_BUFFER, m_vboPos);
	glVertexPointer(4, GL_DOUBLE, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	if (m_vboColor) {
		glBindBuffer(GL_ARRAY_BUFFER, m_vboColor);
		glColorPointer(4, GL_DOUBLE, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);
	}

	if (isTransparency) {
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		//glBlendFunc(GL_ONE, GL_ONE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	glDrawArrays(GL_POINTS, 0, m_numParticles);

	if (isTransparency) {
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);


}
inline float dot(float* a, float* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline double length(float* v)
{
    return sqrt(dot(v, v));
}

void ParticleRenderer::_drawVelLines() {
	float t_pos1[3];
	float t_pos2[3];
	int k = 0;
	int j = 0;

	glBindBuffer(GL_ARRAY_BUFFER, m_vboPos);
	double *pos = (double *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
	glBegin(GL_LINES);
	for (int i = 0; i < m_numParticles; ++i) {
		t_pos1[0] = pos[k++];
		t_pos1[1] = pos[k++];
		t_pos1[2] = pos[k++];

		t_pos2[0] = m_vel[j++];
		t_pos2[1] = m_vel[j++];
		t_pos2[2] = m_vel[j++];

		glVertex3f(t_pos1[0], t_pos1[1], t_pos1[2]);

		if (normalizeVelSpace) {
			double len_vec = length(t_pos2) / m_particleRadius;
			if (len_vec > 1.0) {
				t_pos2[0] = t_pos2[0] / len_vec;
				t_pos2[1] = t_pos2[1] / len_vec;
				t_pos2[2] = t_pos2[2] / len_vec;
			}
		}

		glVertex3f(t_pos1[0] + t_pos2[0],
									t_pos1[1] + t_pos2[1], t_pos1[2] + t_pos2[2]);

		j++;
		k++;
	}
	glEnd();
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
};
void ParticleRenderer::_drawAngleLines()
{
	int k = 0;
	int j = 2;
	float t_pos[3];
	float t_angle;
	glBegin(GL_LINES);
	for (int i = 0; i < m_numParticles; ++i) {
		t_pos[0] = m_pos[k++];
		t_pos[1] = m_pos[k++];
		t_pos[2] = m_particleRadius;
		//t_pos[2] = 0;
		t_angle = m_angles[j];
		glVertex3f(t_pos[0], t_pos[1], t_pos[2]);
		glVertex3f(t_pos[0] + m_particleRadius * sin(t_angle),	t_pos[1] + m_particleRadius * cos(t_angle), t_pos[2]);
		j+=4;
		k+=2;
	}
	glEnd();
}

inline void drawSphere(double r, int lats, int longs) {
	int i, j;
	r = 1.0;
	for (i = 0; i <= lats; i++) {
		double lat0 = M_PI * (-0.5 + (double) (i - 1) / lats);
		double z0 = r * sin(lat0);
		double zr0 = r * cos(lat0);

		double lat1 = M_PI * (-0.5 + (double) i / lats);
		double z1 = r * sin(lat1);
		double zr1 = r * cos(lat1);

		glBegin(GL_QUAD_STRIP);
		for (j = 0; j <= longs; j++) {
			double lng = 2 * M_PI * (double) (j - 1) / longs;
			double x = r * cos(lng);
			double y = r * sin(lng);

			glNormal3f(x * zr0, y * zr0, z0);
			glVertex3f(x * zr0, y * zr0, z0);
			glNormal3f(x * zr1, y * zr1, z1);
			glVertex3f(x * zr1, y * zr1, z1);
		}
		glEnd();
	}
}


void ParticleRenderer::display(ColorMode mode /* = PARTICLE_POINTS */)
{
	// collider
	glPushMatrix();
	glTranslatef(colliderPos.x, colliderPos.y, colliderPos.z);
	glColor3f(1.0, 0.0, 0.0);
	drawSphere(colliderRad, 20, 10);
	//glutSolidSphere(colliderRad, 20, 10);
	glPopMatrix();

	if (mode == COLOR_VELOCITY_SPACE) {
		glColor3f(1, 1, 1);
		glPointSize(m_pointSize);
		_drawPoints();
		_drawVelLines();
	} else {
		glEnable(GL_POINT_SPRITE_ARB);
		glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(m_program);

		glUniform1f(glGetUniformLocation(m_program, "pointScale"),
				m_window_h / tanf(m_fov * 0.5f * (double) M_PI / 180.0f));
		glUniform1f(glGetUniformLocation(m_program, "pointRadius"),
				m_particleRadius);

		glColor3f(1, 1, 1);

		_drawPoints();

		glUseProgram(0);
		glDisable(GL_POINT_SPRITE_ARB);
	}
	//_drawAngleLines();
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif


}
