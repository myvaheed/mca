
//#include "particles_main.h"

#include "gui/qtcudamca.h"

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

#include "core/McaSystem.h"
#include "renderer/render_particles.h"
#include "paramgl.h"

#include <chrono>
#include <thread>

//#define NUM_PARTICLES   16384


#define NUM_PARTICLES   3375
#define NUM_COLUMN   15
#define NUM_ROW   15

#define SPACE_SIZE 240
#define PARTICLE_RADIUS 5


// view params
int ox, oy;
double camera_trans[] = { -SPACE_SIZE/2, 0, -SPACE_SIZE/2 };
double camera_rot[] = { 0, 0, 0 };
//double camera_trans_lag[] = { 0, 0, -3 };
double camera_trans_lag[] = { -SPACE_SIZE/2, 0, -SPACE_SIZE/2 };
double camera_rot_lag[] = { 0, 0, 0 };
const double inertia = 0.1f;


int _controlMode = 0;

bool bPause = true;

enum {
	M_VIEW = 0, M_MOVE
};

uint _numParticles = 0;
uint _spaceSizeWidth = 0;
uint _spaceSizeHeight = 0;
uint _spaceSizeDepth = 0;
uint _spaceSizeMax = 0;

double _particleRadius = 0;
double _particleMass = 0;
uint _numColumn = 0;
uint _numRow = 0;
McaSystem::ParticleConfig _packMode = McaSystem::ParticleConfig::CONFIG_GRID;
McaSystem::ParticleExternalForce _forceMode = McaSystem::EXTERNAL_FORCE_DEFAULT;
ParticleRenderer::ColorMode _displayMode = ParticleRenderer::COLOR_TENSION;

double _mcaE;
double _mcaMu;
double _mcaRmin;
double _mcaRmax;
bool _isBrittle;

volatile bool _isStarted = false;

// simulation parameters
double _timestep;
double _forceVal;
double _colliderSpring;
bool _transparency;

QtCudaMCA *refQtCudaMca = 0;
MainWindow* refMainWindow = 0;
McaSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
static std::chrono::duration<double> secPerFrame;
static double globalSeconds = 0;
static double realtimeSeconds = 0;
static double fps = 0;

StopWatchInterface *timer = NULL;

std::chrono::time_point<std::chrono::system_clock> beginTime, endTime;

ParticleRenderer *renderer = 0;

float modelView[16];

unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;


//Test Mode parameters
bool _GUI_enabled = true;
bool _isTesting = false;
uint _numIterations = 0;
uint _numThreads = 0;
bool _isGPUonTest = false;

const char *txtMCA = "Movable Cellular Automate";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device,
		unsigned int vbo, int size);

bool computationThread();

void init_camera() {
	camera_trans[0] = 0;
	camera_trans[1] = 0;
	camera_trans[2] = 0;

	camera_rot[0] = 0;
	camera_rot[1] = 0;
	camera_rot[2] = 0;

	camera_trans_lag[0] = 0;
	camera_trans_lag[1] = 0;
    camera_trans_lag[2] = 0;

	camera_rot_lag[0] = 0;
	camera_rot_lag[1] = 0;
	camera_rot_lag[2] = 0;
}
// initialize particle system
void initParticleSystem(int numParticles, double particleRadius, double mass, double space_width, double space_height, double space_depth) {

	psystem = new McaSystem(numParticles, particleRadius, mass, space_width, space_height, space_depth,
			_GUI_enabled);

	psystem->setTimestep(_timestep);
	psystem->setExternalForceVal(_forceVal);
	psystem->setCollideSpring(_colliderSpring);
	psystem->setMcaConstModulus(_mcaRmin, _mcaRmax, _mcaE, _mcaMu, _isBrittle);

	psystem->configSystem(_numColumn, _numRow, _packMode, _forceMode);

	if (_GUI_enabled) {
		renderer = new ParticleRenderer();
		renderer->setParticleRadius(particleRadius);

		renderer->setData(psystem->getPositionBuffer(),
				psystem->getColorLinksBuffer(), psystem->getColorDeformBuffer(),
				psystem->getColorAccelBuffer(), psystem->getNumParticles());
		renderer->setColorMode(_displayMode);
		renderer->setIsTransparancy(_transparency);
	}

	sdkCreateTimer(&timer);

	frameCount = 0;
	fpsCount = 0;
	fpsLimit = 1;
	globalSeconds = 0;
	realtimeSeconds = 0;
	fps = 0;

	_controlMode = M_VIEW;

	init_camera();

	beginTime = std::chrono::system_clock::now();
}


void cleanup() {
	sdkDeleteTimer(&timer);

	if (psystem) {
		delete psystem;
		psystem = 0;
	}

	if (renderer) {
		delete renderer;
		renderer = 0;
	}

	return;
}

bool started() {
	printf("started\n");
	_isStarted = true;
	initParticleSystem(_numParticles, _particleRadius, _particleMass, _spaceSizeWidth, _spaceSizeHeight, _spaceSizeDepth);
	return true;
}

bool stopped() {
	printf("stopped\n");
	_isStarted = false;
	cleanup();
	return true;
}


void MainWindow::updatedEnv(int Nparts, int countOfCol, int countOfRow, double automateD, double automateM, UIPackMode mode ) {
	if (countOfCol == 0 || countOfRow == 0)
		return;
	_numParticles = Nparts;
	_numColumn = countOfCol;
	_numRow = countOfRow;
	//_particleRadius = automateD / sqrt(3.0);

	_particleRadius = automateD / 2.0;
	_particleMass = automateM;

	printf("mass = %f\n", _particleMass);

	switch (mode) {
	case UIPackMode::PACK_GRID: {
		_packMode = McaSystem::ParticleConfig::CONFIG_GRID;
		_spaceSizeWidth = 2 * _particleRadius * _numColumn * (1 + 2);
		_spaceSizeHeight = 2 * _particleRadius * (_numParticles / _numColumn + 1) * (1 + 2);
		_spaceSizeDepth = 2 * _particleRadius * (1 + 2);
	}
		break;
	case UIPackMode::PACK_SQUARE: {
		_packMode = McaSystem::ParticleConfig::CONFIG_SQUARE;
		_spaceSizeWidth = 2 * _particleRadius * _numColumn * (1 + 2);
		_spaceSizeHeight = 2 * _particleRadius
				* (_numParticles / _numColumn + 1) * (1 + 2);
		_spaceSizeDepth = 2 * _particleRadius * (1 + 2);
	}
		break;
	case UIPackMode::PACK_HCP: {
		_packMode = McaSystem::ParticleConfig::CONFIG_HCP;
		_spaceSizeWidth = 2 * _particleRadius * _numColumn * (1 + 0.8);
		_spaceSizeHeight = 2 * _particleRadius * _numRow * (1 + 0.8);
		_spaceSizeDepth = 2 * _particleRadius * (_numParticles / (_numRow * _numColumn) + 1) * (1 + 0.5);
	}
		break;
	}

	_spaceSizeMax = std::max(std::max(_spaceSizeWidth, _spaceSizeHeight), _spaceSizeDepth);

	updateConst();
}

void MainWindow::updatedConst(double rMin, double rMax, double mcaE, double mcaMu, double mcaViscosity, bool isBrittle) {
	_mcaE = mcaE;
	_mcaMu = mcaMu;
	_mcaRmax = _particleRadius * rMax / 100.0;
	_mcaRmin = _particleRadius * rMin / 100.0;
	_isBrittle = isBrittle;
}

void MainWindow::updatedRunMode(double timestep, double deltaForce, double colliderSpring, bool transparency, UIForceMode forceMode, UIColorMode colorMode) {
	_isStarted = false;
	_timestep = timestep;

#if SOFTFVAL

#else
	_forceVal = deltaForce;
#endif

	_colliderSpring = colliderSpring;
	_transparency = transparency;

	switch(forceMode) {
	case UIForceMode::EXTERNAL_FORCE_DEFAULT : {
		_forceMode = McaSystem::EXTERNAL_FORCE_DEFAULT;
	}
		break;
	case UIForceMode::EXTERNAL_FORCE_CORNER: {
		_forceMode = McaSystem::EXTERNAL_FORCE_CORNER;
	}
		break;
	case UIForceMode::EXTERNAL_FORCE_TWO_CORNERS: {
		_forceMode = McaSystem::EXTERNAL_FORCE_TWO_CORNERS;
	}
		break;
	case UIForceMode::EXTERNAL_FORCE_FORCER: {
		_forceMode = McaSystem::EXTERNAL_FORCE_FORCER;
	}
		break;
	case UIForceMode::EXTERNAL_FORCE_PULL_APART: {
		_forceMode = McaSystem::EXTERNAL_FORCE_PULL_APART;
	}
		break;
	case UIForceMode::EXTERNAL_FORCE_PUSH_APART: {
		_forceMode = McaSystem::EXTERNAL_FORCE_PUSH_APART;
	}
		break;
	}

	switch (colorMode) {
	case UIColorMode::COLOR_ACCEL: {
		_displayMode = ParticleRenderer::COLOR_TENSION;
	}
		break;
	case UIColorMode::COLOR_DEFORM: {
		_displayMode = ParticleRenderer::COLOR_DEFORM;
	}
		break;
	case UIColorMode::COLOR_LINKS: {
		_displayMode = ParticleRenderer::COLOR_LINKS;
	}
		break;
	case UIColorMode::COLOR_VELOCITY_SPACE: {
		_displayMode = ParticleRenderer::COLOR_VELOCITY_SPACE;
	}
		break;
	}

	if (!psystem || !renderer)
		return;

	psystem->setTimestep(_timestep);
	psystem->setExternalForceVal(_forceVal);
	psystem->setCollideSpring(_colliderSpring);
	psystem->setExternalForceMode(_numColumn, _numRow, _forceMode);
	renderer->setColorMode(_displayMode);
	renderer->setIsTransparancy(_transparency);
	if (_displayMode == ParticleRenderer::COLOR_VELOCITY_SPACE) {
		renderer->setData(
				(double *) psystem->getArray(McaSystem::POSITION), 0,
				(double *) psystem->getArray(McaSystem::VELOCITY), 0, 0, 0,
				psystem->getNumParticles(), 0);
	}
	_isStarted = true;
}

void MainWindow::updatedTest(UIHardwareType hardwareType, int numThreads, int numIterations) {
	_isGPUonTest = hardwareType == UIHardwareType::HARDWARE_GPU ? true : false;
	_numThreads = numThreads;
	_numIterations = numIterations;
}

bool MainWindow::startClicked() {
	if (!_isStarted) {
		bPause = false;
		return true;
	} else {
		if (bPause) {
			bPause = false;
			return true;
		}
		bPause = true;
		return false;
	}
}

bool MainWindow::stopClicked() {
	if (_isStarted) {
		bPause = true;
		stopped();
		return true;
	}
	else
		return false;
}


bool MainWindow::resetClicked() {
	if (_isStarted)
		return true;
	return false;
}

bool MainWindow::testClicked() {
	if (_isTesting)
		return false;
	_isTesting = true;
	bPause = false;
	return true;
}

bool MainWindow::stopTestClicked() {
	if(!_isTesting)
		return false;
	_isTesting = false;
	bPause = true;
	return true;
}


void computeFPS() {
	endTime = std::chrono::system_clock::now();
	secPerFrame = endTime - beginTime;
	beginTime = std::chrono::system_clock::now();

	if (bPause)
		return;

	globalSeconds += secPerFrame.count();
	realtimeSeconds += _timestep;
	frameCount++;
	fpsCount++;
	if (fpsCount == fpsLimit) {
		double ifps = 1.f / secPerFrame.count();
		fps = ifps;
		fpsCount = 0;
		fpsLimit = (int) MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}
}

inline double frand() {
	return rand() / (double) RAND_MAX;
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


void McaOGLWindow::initializeGL() {
	printf ("initGL\n");
	initializeOpenGLFunctions();
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

	cleanup();
	started();
}

void McaOGLWindow::resizeGL(int w, int h) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(60.0, (double) w / (double) h, 0.1, 1000 * 100);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, w, h);

	if (renderer) {
		renderer->setWindowSize(w, h);
		renderer->setFOV(60.0);
	}
}

bool computationThread() {
	if (!_isStarted) {
		printf("it is not started\n");
		return false;
	}


	if (_isTesting) {
		//printf("testing..\n");
		psystem->updateVelretTestMode(_isGPUonTest, _numThreads);
		computeFPS();
		printf("iter frameCount=%d, numIter=%d\n", frameCount, _numIterations);
		if (frameCount >= _numIterations) {
			printf("test finished: %f, numThread: %d\n", globalSeconds, _numThreads);
			psystem->dumpParticles(_numParticles - 30, 20);
			if (_GUI_enabled) {
				refMainWindow->testFinished(globalSeconds);
			}
			return false;
		}
		if (_GUI_enabled)
			refMainWindow->glGUIDrawStep(realtimeSeconds, globalSeconds, fps);
		return true;
	}

#if SOFTFVAL
	if (frameCount <= 10000) {
		_forceVal += 0.1;
		printf("forceVal = %f\n", _forceVal);
		psystem->setExternalForceVal(_forceVal);
	}

	if (frameCount % 100 == 0) {
		psystem->writeSelectedArrayToFile("stress.txt", McaSystem::ParticleArray::STRESS);
	}

#endif

	sdkStartTimer(&timer);
	if (!bPause) {
		psystem->updateVerlet();
		if (renderer && _displayMode == ParticleRenderer::COLOR_VELOCITY_SPACE) {
			renderer->setData((double *) psystem->getArray(McaSystem::POSITION),
					0, (double *) psystem->getArray(McaSystem::VELOCITY), 0, 0,
					0, psystem->getNumParticles(), 0);

		}
	}

	sdkStopTimer(&timer);
	computeFPS();


	return true;
}

void McaOGLWindow::paintGL() {
	if (!computationThread())
		return;
	refMainWindow->glGUIDrawStep(realtimeSeconds, globalSeconds, fps);
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

	if (renderer) {
		renderer->setColliderPosRad(psystem->getColliderPos(),
				psystem->getColliderRadius());

		renderer->display(_displayMode);
	}

	update();
}

void McaOGLWindow::mousePressed(int x, int y) {
	ox = x;
	oy = y;
}

void McaOGLWindow::leftButtonMotion(int x, int y) {
	double dx, dy;
	dx = (double) (x - ox);
	dy = (double) (y - oy);
	switch (_controlMode) {
	case M_VIEW:
		camera_rot[0] += dy / 5.0f;
		camera_rot[1] += dx / 5.0f;
		break;

	case M_MOVE: {
		double translateSpeed = 0.03f;
		double3 p = psystem->getColliderPos();

		float v[3], r[3];
		v[0] = _spaceSizeMax / 50.0 * dx * translateSpeed;
		v[1] = _spaceSizeMax / 50.0 * -dy * translateSpeed;
		v[2] = 0.0f;
		ixform(v, r, modelView);
		p.x += r[0];
		p.y += r[1];
		p.z += r[2];

		psystem->setColliderPos(p);
	}
		break;
	}
	ox = x;
	oy = y;
}

void McaOGLWindow::middleButtonMotion(int x, int y) {
	double dx, dy;
	dx = (double) (x - ox);
	dy = (double) (y - oy);
	switch (_controlMode) {
	case M_VIEW:
		camera_trans[0] += _spaceSizeMax/2 * dx / 100.0f;
		camera_trans[1] -= _spaceSizeMax/2 * dy / 100.0f;

		break;

	case M_MOVE: {
		double translateSpeed = 0.03f;
		double3 p = psystem->getColliderPos();

		float v[3], r[3];
		v[0] = 0.0f;
		v[1] = 0.0f;
		v[2] = dy * translateSpeed;
		ixform(v, r, modelView);
		p.x += r[0];
		p.y += r[1];
		p.z += r[2];

		psystem->setColliderPos(p);
	}

		break;
	}
	ox = x;
	oy = y;
}

void print_double3_2(const char* txt, double3 vec) {
	printf("%s: %3.6f, %3.6f, %3.6f\n", txt, vec.x, vec.y, vec.z);
}

void McaOGLWindow::rightButtonClicked(int x, int y) {
	double matModelView[16], matProjection[16];
	int viewport[4];
	glGetDoublev( GL_MODELVIEW_MATRIX, matModelView);
	glGetDoublev( GL_PROJECTION_MATRIX, matProjection);
	glGetIntegerv( GL_VIEWPORT, viewport);
	double3 m_start;
	double3 m_end;
	double winX = (double) x;
	double winY = viewport[3] - (double) y;
	gluUnProject(winX, winY, 0.0, matModelView, matProjection, viewport,
			&m_start.x, &m_start.y, &m_start.z);
	gluUnProject(winX, winY, 1.0, matModelView, matProjection, viewport,
			&m_end.x, &m_end.y, &m_end.z);
	print_double3_2("start: ", m_start);
	print_double3_2("end: ", m_end);
	psystem->selectParticle(m_start, m_end);
}


void McaOGLWindow::rightButtonMotion(int x, int y) {

}

void McaOGLWindow::zoomButtonMotion(bool zoom) {
	if (zoom) {
		camera_trans[2] += _spaceSizeMax/2 * 0.8f;
	} else {
		camera_trans[2] -= _spaceSizeMax/2 * 0.8f;
	}
}

void McaOGLWindow::spacePressed() {
	bPause = !bPause;
}

void McaOGLWindow::V_Pressed() {
	_controlMode = M_VIEW;
}

void McaOGLWindow::M_Pressed() {
	_controlMode = M_MOVE;
}

void QtCudaMCA::getMainWindow(MainWindow* refW) {
	refMainWindow = refW;
}

int main( int   argc,
          char *argv[] )
{

	if (argc > 1) {
		if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
			printf(
					"sample:\n \
cuda_particles -npart=3375 -test -nth=256 -asize=5 -rmax=20 \
-rmin=-10 -e=70000 -mu=0.34 -ncol=15 -step=0.0001 -fval=600 -fmod=2 -niter=100\n");
			return 0;
		}
	}


#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
	setenv("CUDA_DEBUGGER_SOFTWARE_PREEMPTION", "1", 0);
#endif

	printf("%s Starting...\n\n", txtMCA);

	_numParticles = NUM_PARTICLES;
	_spaceSizeMax = SPACE_SIZE;
	_particleRadius = PARTICLE_RADIUS;
	_numColumn = NUM_COLUMN;
	_numRow = NUM_ROW;

	if (argc > 1) {

		if (checkCmdLineFlag(argc, (const char **) argv, "npart")) {
			_numParticles = getCmdLineArgumentInt(argc, (const char **) argv,
					"npart");
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "ngui")) {
			_GUI_enabled = false;
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "test")) {
			_isTesting = true;
		}

		try {
			if (_isTesting) {
				_packMode = McaSystem::ParticleConfig::CONFIG_GRID;
				_GUI_enabled = false;
				_numThreads = getCmdLineArgumentInt(argc, (const char **) argv,
						"nth");
				_particleRadius = getCmdLineArgumentFloat(argc,
						(const char **) argv, "asize") / 2.0;
				_particleMass = getCmdLineArgumentFloat(argc, (const char **) argv, "m");

				_mcaRmax = _particleRadius * getCmdLineArgumentFloat(argc, (const char **) argv,
						"rmax") / 100.0;
				_mcaRmin = _particleRadius * getCmdLineArgumentFloat(argc, (const char **) argv,
						"rmin") / 100.0;
				_mcaE = getCmdLineArgumentFloat(argc, (const char **) argv,
						"e");
				_mcaMu = getCmdLineArgumentFloat(argc, (const char **) argv,
						"mu");

				_numColumn = getCmdLineArgumentInt(argc, (const char **) argv,
						"ncol");

				_timestep = getCmdLineArgumentFloat(argc, (const char **) argv,
						"step");

				_forceVal = getCmdLineArgumentFloat(argc, (const char **) argv,
						"fval");

				_numIterations = getCmdLineArgumentInt(argc,
						(const char **) argv, "niter");

				_isGPUonTest = !checkCmdLineFlag(argc, (const char **) argv,
						"cpu");
				_forceMode =
						(McaSystem::ParticleExternalForce) getCmdLineArgumentInt(
								argc, (const char **) argv, "fmod");

				_spaceSizeWidth = 2 * _particleRadius * _numColumn * (1 + 1);
				_spaceSizeHeight = 2 * _particleRadius
						* (_numParticles / _numColumn + 1) * (1 + 1);
				_spaceSizeDepth = 2 * _particleRadius * (1 + 1);

				_spaceSizeMax = std::max(
						std::max(_spaceSizeWidth, _spaceSizeHeight),
						_spaceSizeDepth);
			}
		} catch (...) {
			printf(
					"sample:\n \
cuda_particles -npart=3375 -test -nth=256 -asize=5 -rmax=20 \n \
-rmin=-10 -e=70000 -mu=0.34 -ncol=15 -step=0.0001 -fval=600 -fmod=2 -niter=100 \n");
			return -1;
		}
	}

	printf("particles: %d\n", _numParticles);
	printf("forceMode=%s\n", UIForceModeStrings[_forceMode]);
	printf("fVal=%f\n", _forceVal);
	printf("numThread=%d\n", _numThreads);
	printf("numCol=%d\n", _numColumn);
	printf("isGPU=%d\n", _isGPUonTest);
	printf("timestep=%f\n", _timestep);
	printf("rmax=%f\n", _mcaRmax);
	printf("rmin=%f\n", _mcaRmin);
	printf("mu=%f\n", _mcaMu);
	printf("e=%f\n", _mcaE);
	printf("rad=%f\n", _particleRadius);

	cudaInit(argc, argv);
	if (_GUI_enabled)
		cudaGLInit(argc, argv);

	int result;

	if (_GUI_enabled) {
		QtCudaMCA qtCudaMca;
		refQtCudaMca = &qtCudaMca;
		result = qtCudaMca.run(argc, argv);
		cleanup();
	} else {
		_isStarted = true;
		bPause = false;
		initParticleSystem(_numParticles, _particleRadius, _particleMass, _spaceSizeWidth, _spaceSizeHeight, _spaceSizeDepth);
		while (true) {
			if (!computationThread())
				break;
		}
		cleanup();
	}
	printf("result = %d\n", result);

	exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

