/*

#include "particles_main.h"

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"

#include <chrono>

//#define NUM_PARTICLES   16384


#define NUM_PARTICLES   3375
#define NUM_COLUMN   15
#define NUM_ROW   15

#define SPACE_DIM 2
#define SPACE_SIZE 240
#define PARTICLE_RADIUS 5



#define NUM_PARTICLES   16
#define NUM_COLUMN   4

#define SPACE_DIM 200

#define PARTICLE_RADIUS 5



const uint width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
double camera_trans[] = { -SPACE_SIZE/2, 0, -SPACE_SIZE/2 };
double camera_rot[] = { 0, 0, 0 };
double camera_trans_lag[] = { 0, 0, -3 };
double camera_rot_lag[] = { 0, 0, 0 };
const double inertia = 0.1f;
ParticleRenderer::ColorMode displayMode = ParticleRenderer::COLOR_TENSION;

int mode = 0;
bool displayEnabled = true;
//bool bPause = false;
bool bPause = true;
bool displaySliders = false;

enum {
	M_VIEW = 0, M_MOVE
};

uint numParticles = 0;
uint numColumn = 0;
uint numRow = 0;
int numIterations = 0; // run until exit

// simulation parameters
double timestep = 0.5f;
double gravity = 0.0003f;
double deltaForce = 0.0f;
//double gravity = 1.0f;

int iterations = 1;

double collideSpring = 0.2f;
double collideShear = 0.1f;


ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
static std::chrono::duration<double> secPerFrame;
static double globalSeconds = 0;
StopWatchInterface *timer = NULL;

std::chrono::time_point<std::chrono::system_clock> beginTime, endTime;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

const char *txtMCA_sample = "Movable Cellular Automate";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device,
		unsigned int vbo, int size);

// initialize particle system
void initParticleSystem(int numParticles, double particleRadius, double space_width, double space_height) {
	psystem = new ParticleSystem(numParticles, particleRadius, SPACE_SIZE, SPACE_SIZE, SPACE_SIZE);
	psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_GRID, ParticleSystem::NONE);

	renderer = new ParticleRenderer(SPACE_SIZE, 0);
	renderer->setParticleRadius(psystem->getParticleRadius());

	renderer->setData(psystem->getPositionBuffer(), psystem->getColorLinksBuffer(), psystem->getColorDeformBuffer(), psystem->getColorAccelBuffer(),
						psystem->getNumParticles());
	renderer->setColorMode(displayMode);

	beginTime = std::chrono::system_clock::now();
	sdkCreateTimer(&timer);
}

void cleanup() {
	sdkDeleteTimer(&timer);

	if (psystem) {
		delete psystem;
	}
	return;
}

// initialize OpenGL
void initGL(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow(txtMCA_sample);

	if (!isGLVersionSupported(2, 0)
			|| !areGLExtensionsSupported(
					"GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(EXIT_FAILURE);
	}

#if defined (WIN32)

	if (wglewIsSupported("WGL_EXT_swap_control"))
	{
		// disable vertical sync
		wglSwapIntervalEXT(0);
	}

#endif

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.25, 0.25, 0.25, 1.0);

	glutReportErrors();
}

void computeFPS() {
	endTime = std::chrono::system_clock::now();
	secPerFrame = endTime - beginTime;
	beginTime = std::chrono::system_clock::now();

	if (bPause)
		return;

	//secPerFrame = sdkGetAverageTimerValue(&timer) / 1000.f;

	globalSeconds += secPerFrame.count();

	frameCount++;
	fpsCount++;
	if (fpsCount == fpsLimit) {
		char fps[256];

		double ifps = 1.f / secPerFrame.count();

		sprintf(fps, "MCA (%d particles): %3.0f fps %1.1f sec", numParticles,
				ifps, globalSeconds);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int) MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

inline void draw_rectangle_parallepiped() {
	// cube
	glColor3f(1.0, 1.0, 1.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_QUADS);
	double z = PARTICLE_RADIUS / SPACE_SIZE;
	//Top face (y = 1.0f)
	glVertex3f(1.0f, 1.0f, -z);
	glVertex3f(-1.0f, 1.0f, -z);
	glVertex3f(-1.0f, 1.0f, z);
	glVertex3f(1.0f, 1.0f, z);
	// Left face (x = -1.0f)
	glVertex3f(-1.0f, 1.0f, z);
	glVertex3f(-1.0f, 1.0f, -z);
	glVertex3f(-1.0f, -1.0f, -z);
	glVertex3f(-1.0f, -1.0f, z);
	// Front face  (z = 1.0f)
	glVertex3f(1.0f, 1.0f, z);
	glVertex3f(-1.0f, 1.0f, z);
	glVertex3f(-1.0f, -1.0f, z);
	glVertex3f(1.0f, -1.0f, z);
	// Bottom face (y = -1.0f)
	glVertex3f(1.0f, -1.0f, z);
	glVertex3f(-1.0f, -1.0f, z);
	glVertex3f(-1.0f, -1.0f, -z);
	glVertex3f(1.0f, -1.0f, -z);

	// Back face (z = -1.0f)
	glVertex3f(1.0f, -1.0f, -z);
	glVertex3f(-1.0f, -1.0f, -z);
	glVertex3f(-1.0f, 1.0f, -z);
	glVertex3f(1.0f, 1.0f, -z);

	// Right face (x = 1.0f)
	glVertex3f(1.0f, 1.0f, -z);
	glVertex3f(1.0f, 1.0f, z);
	glVertex3f(1.0f, -1.0f, z);
	glVertex3f(1.0f, -1.0f, -z);

	glEnd();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glutWireCube(2.0);
}

void display() {

	sdkStartTimer(&timer);
	// update the simulation
	if (!bPause) {
		psystem->setTimestep(timestep);
		psystem->setExternalDeltaForce(deltaForce);
		psystem->setCollideSpring(collideSpring);
		psystem->setCollideShear(collideShear);
		//psystem->update(timestep * secPerFrame.count()); //realTime
		psystem->updateVerlet(timestep);
		//psystem->updateEiler(timestep);
		if (displayMode == ParticleRenderer::COLOR_VELOCITY_SPACE) {
			renderer->setData((double *) psystem->getArray(ParticleSystem::POSITION), 0, (double *) psystem->getArray(ParticleSystem::VELOCITY), 0, 0, 0, psystem->getNumParticles(), 0);
		}
		//renderer->setColorMode(displayMode);
	}

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// view transform
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	for (int c = 0; c < 3; ++c) {
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c])
				* inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}

	glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
	glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

	//draw_rectangle_parallepiped();

	if (renderer && displayEnabled) {
		renderer->setColliderPosRad(psystem->getColliderPos(), psystem->getColliderRadius());

		renderer->display(displayMode);
	}

	if (displaySliders) {
		glDisable(GL_DEPTH_TEST);
		glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
		glEnable(GL_BLEND);
		params->Render(0, 0);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}

	sdkStopTimer(&timer);

	glutSwapBuffers();
	glutReportErrors();

	computeFPS();
}

inline double frand() {
	return rand() / (double) RAND_MAX;
}


void reshape(int w, int h) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(60.0, (double) w / (double) h, 0.1, SPACE_SIZE * 100);

	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer) {
		renderer->setWindowSize(w, h);
		renderer->setFOV(60.0);
	}
}

void mouse(int button, int state, int x, int y) {
	int mods;

	if (state == GLUT_DOWN) {
		buttonState |= 1 << button;
	} else if (state == GLUT_UP) {
		buttonState = 0;
	}

	mods = glutGetModifiers();

	if (mods & GLUT_ACTIVE_SHIFT) {
		buttonState = 2;
	} else if (mods & GLUT_ACTIVE_CTRL) {
		buttonState = 3;
	}

	if ((button == 3) || (button == 4)) {
		if (state == GLUT_UP)
			return;
		buttonState = 4; // zoom
		if (button == 3) {
			camera_trans[2] += SPACE_SIZE * 0.13f;
		} else {
			camera_trans[2] -= SPACE_SIZE * 0.13f;
		}
	}

	ox = x;
	oy = y;

	if (displaySliders) {
		if (params->Mouse(x, y, button, state)) {
			glutPostRedisplay();
			return;
		}
	}



	glutPostRedisplay();
}

// transform vector by matrix
void xform(double *v, double *r, GLfloat *m) {
	r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
	r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
	r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m) {
	r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
	r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
	r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m) {
	float x[4];
	x[0] = v[0] - m[12];
	x[1] = v[1] - m[13];
	x[2] = v[2] - m[14];
	x[3] = 1.0f;
	ixform(x, r, m);
}

void motion(int x, int y) {
	double dx, dy;
	dx = (double) (x - ox);
	dy = (double) (y - oy);

	if (displaySliders) {
		if (params->Motion(x, y)) {
			ox = x;
			oy = y;
			glutPostRedisplay();
			return;
		}
	}

	switch (mode) {
	case M_VIEW:
		if (buttonState & 2) {
			// middle = translate
			camera_trans[0] += SPACE_SIZE * dx / 100.0f;
			camera_trans[1] -= SPACE_SIZE * dy / 100.0f;
		} else if (buttonState & 1) {
			// left = rotate
			camera_rot[0] += dy  / 5.0f;
			camera_rot[1] += dx / 5.0f;
		}

		break;

	case M_MOVE: {
		double translateSpeed = 0.03f;
		double3 p = psystem->getColliderPos();

		if (buttonState == 1) {
			float v[3], r[3];
			v[0] = SPACE_SIZE/10.0 * dx * translateSpeed;
			v[1] = SPACE_SIZE/10.0 * -dy * translateSpeed;
			v[2] = 0.0f;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		} else if (buttonState == 2) {
			float v[3], r[3];
			v[0] = 0.0f;
			v[1] = 0.0f;
			v[2] = dy * translateSpeed;
			ixform(v, r, modelView);
			p.x += r[0];
			p.y += r[1];
			p.z += r[2];
		}

		psystem->setColliderPos(p);
	}
		break;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int x, int y) {
	switch (key) {
	case ' ':
		bPause = !bPause;
		break;

	case '\033':
	case 'q':
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	case 'v':
		mode = M_VIEW;
		break;

	case 'm':
		mode = M_MOVE;
		break;

	case 'p':
		displayMode = (ParticleRenderer::ColorMode) ((displayMode + 1)
				% ParticleRenderer::COLOR_NUM_MODES);
		break;

	case 'd':
		psystem->dumpGrid();
		break;

	case 'u':
		psystem->dumpParticles(0, numParticles - 1);
		break;

	case 'r':
		displayEnabled = !displayEnabled;
		break;

	case '1':
		psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_HCP, ParticleSystem::NONE);
		break;

	case '2':
		psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_HCP, ParticleSystem::EXTERNAL_FORCE_CORNER);
				break;
	case '3':
		psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_HCP, ParticleSystem::EXTERNAL_FORCE_FORCER);
						break;
	case '4':
		psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_HCP,
								ParticleSystem::EXTERNAL_FORCE_PUSH_APART);
		break;
	case '5':
		psystem->configSystem(numColumn, numRow, ParticleSystem::ParticleConfig::CONFIG_HCP,
								ParticleSystem::EXTERNAL_FORCE_PULL_APART);
		break;
	case '6':
		psystem->configSystem(numColumn, numRow,
						ParticleSystem::ParticleConfig::CONFIG_GRID,
						ParticleSystem::NONE);
		break;
	case '7':
		psystem->configSystem(numColumn, numRow,
						ParticleSystem::ParticleConfig::CONFIG_GRID,
						ParticleSystem::EXTERNAL_FORCE_FORCER);
		break;
	case '8':
		psystem->configSystem(numColumn, numRow,
						ParticleSystem::ParticleConfig::CONFIG_GRID,
						ParticleSystem::EXTERNAL_FORCE_CORNER);
		break;
	case '9':
		psystem->configSystem(numColumn, numRow,
						ParticleSystem::ParticleConfig::CONFIG_GRID,
						ParticleSystem::EXTERNAL_FORCE_PULL_APART);
		break;
	case 'h':
		displaySliders = !displaySliders;
		break;
	case 't':
			renderer->setIsTransparancy(!renderer->getIsTransparancy());
			break;
	}
	glutPostRedisplay();
}

void special(int k, int x, int y) {
	if (displaySliders) {
		params->Special(k, x, y);
	}
}

void idle(void) {
	glutPostRedisplay();
}

void initParams() {
	// create a new parameter list
	params = new ParamListGL("misc");
	params->AddParam(
			new Param<double>("time step", timestep, 0.000001, 0.0001, 0.000001,
					&timestep));
	params->AddParam(
				new Param<double>("ex delta force", deltaForce, 0.0, 1000000000.0, 0.001,
						&deltaForce));


	params->AddParam(
				new Param<double>("time step", timestep, 0.0001, 0.01, 0.0001,
						&timestep));
	params->AddParam(
				new Param<double>("ex delta force", deltaForce, 0.0, 1000.0, 0.01,
						&deltaForce));
	params->AddParam(
			new Param<double>("gravity", gravity, 0.0f, 1.0f, 0.0001f,
					&gravity));

	params->AddParam(
				new Param<double>("collide spring", collideSpring, 0.002f, 1000.0f,
						0.001f, &collideSpring));
	params->AddParam(
			new Param<double>("collide shear", collideShear, 0.0f, 1.0f, 0.001f,
					&collideShear));
}

void mainMenu(int i) {
	key((unsigned char) i, 0, 0);
}

void initMenus() {
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Reset block [1]", '1');
	glutAddMenuEntry("Reset random [2]", '2');
	glutAddMenuEntry("Add sphere [3]", '3');
	glutAddMenuEntry("View mode [v]", 'v');
	glutAddMenuEntry("Move cursor mode [m]", 'm');
	glutAddMenuEntry("Toggle point rendering [p]", 'p');
	glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Step animation [ret]", 13);
	glutAddMenuEntry("Toggle sliders [h]", 'h');
	glutAddMenuEntry("Quit (esc)", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
	setenv("CUDA_DEBUGGER_SOFTWARE_PREEMPTION", "1", 0);
#endif

	printf("%s Starting...\n\n", txtMCA_sample);

	printf(
			"NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");

	numParticles = NUM_PARTICLES;
	numColumn = NUM_COLUMN;
	numRow = NUM_ROW;
	numIterations = 0;

	if (argc > 1) {
		if (checkCmdLineFlag(argc, (const char **) argv, "n")) {
			numParticles = getCmdLineArgumentInt(argc, (const char **) argv,
					"n");
		}
	}

	printf("particles: %d\n", numParticles);

	if (checkCmdLineFlag(argc, (const char **) argv, "i")) {
		numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "device")) {
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n in OpenGL mode\n");
		printf(
				"   To use -device=n, the sample must be running w/o OpenGL\n\n");
		printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	initGL(&argc, argv);
	cudaGLInit(argc, argv);

	initParticleSystem(numParticles, PARTICLE_RADIUS, SPACE_SIZE, SPACE_SIZE);

	initParams();

	initMenus();


	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutSpecialFunc(special);
	glutIdleFunc(idle);

	glutCloseFunc(cleanup);

	glutMainLoop();

	if (psystem) {
		delete psystem;
	}

	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}


*/
