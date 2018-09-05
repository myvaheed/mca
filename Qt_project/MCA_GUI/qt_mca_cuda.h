#ifndef LIBRARY_HELPER_H
#define LIBRARY_HELPER_H
enum UIForceMode {
    EXTERNAL_FORCE_DEFAULT = 0,
    EXTERNAL_FORCE_CORNER,
    EXTERNAL_FORCE_TWO_CORNERS,
    EXTERNAL_FORCE_FORCER,
    EXTERNAL_FORCE_PUSH_APART,
    EXTERNAL_FORCE_PULL_APART,
    UI_FORCEMODE_SIZE
};


static const char* UIForceModeStrings[] = {"Default", "Right corner", "Corners", "Forcer", "Push apart", "Pull apart"};

enum UIPackMode {
    PACK_GRID = 0,
    PACK_HCP,
    UI_PACK_MODE_SIZE
};

static const char* UIPackModeStrings[] = {"2D", "3D"};

enum UIColorMode {
    COLOR_LINKS = 0,
    COLOR_DEFORM,
    COLOR_ACCEL,
    COLOR_VELOCITY_SPACE,
    UI_COLOR_MODE_SIZE
};
static const char* UIColorModeStrings[] = {"Links", "Deform", "Tension", "VelocitySpace"};

enum UIHardwareType {
    HARDWARE_GPU = 0,
    HARDWARE_CPU,
    UI_HARDWARE_TYPE_SIZE
};
static const char* UIHardwareTypeStrings[] = { "GPU", "CPU"};

enum UINumThreadsCPU {
    THREADS_CPU_ALL = 0,
    THREADS_CPU_1,
    THREADS_CPU_2,
    THREADS_CPU_4,
    UI_NUM_THREADS_CPU_SIZE
};
static const char* UINumThreadsCPUStrings[] = {"All", "1", "2", "4"};

enum UINumThreadsGPU {
    THREADS_GPU_ALL = 0,
    THREADS_GPU_16,
    THREADS_GPU_32,
    THREADS_GPU_64,
    THREADS_GPU_128,
    THREADS_GPU_256,
    THREADS_GPU_512,
    THREADS_GPU_1024,
    THREADS_GPU_2048,
    THREADS_GPU_5096,
    UI_NUM_THREADS_GPU_SIZE
};
static const char* UINumThreadsGPUStrings[] = {"All", "16", "32", "64", "128", "256", "512", "1024", "2048", "5096"};

#endif // LIBRARY_HELPER_H
