#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "Qt/qmainwindow.h"
#include "qt_defines.h"
#include "mcaoglwidget.h"
#include "testform.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void glDraw();
    void glGUIDrawStep(double realtimeSeconds, double runtimeSeconds, double fps);
    void testFinished(double timeProcessed);
    //DEFINE THESE METHODS!!!
    //input
    void updatedEnv(int Nparts, int countOfCol, int countOfRow, double automateD, double automateM, UIPackMode confMode);
    void updatedConst(double mcaRminPercent, double mcaRmaxPercent, double mcaE, double mcaMu, double mcaViscosity, bool isBrittle);
    void updatedRunMode(double timestep, double deltaForce, double collierSpring, bool transparency, UIForceMode forceMode, UIColorMode colorMode);
    void updatedTest(UIHardwareType hardwareType, int numThreads, int numIterations);

    bool startClicked();
    bool stopClicked();
    bool resetClicked();
    bool testClicked();
    bool stopTestClicked();

    //output
    double getAutomateVolume();
    double getTotalMass();
    double getTotalVolume();
    double getDensity();
    double getShearModulus();
    double getBulkModulus();

    int getNumThreads(UIHardwareType hardwareType);
public slots:
    void updateEnv();
    void updateAutD();
    void updateConst();
    void updateRun();
    void updateTest();
    void startClick();
    void stopClick();
    void resetClick();
    void testClick();
    void stopTestClick();

    void hardwareTypeChanged(int value);
    void sliderTimestepChanged(int value);
    void sliderDeltaForceChanged(int value);
    void sliderColliderSpringChanged(int value);

private:

    void onPause();
    void onStart();
    void onStop();
    void onTestStart();
    void onTestStop();
    void initGUI();
    bool isRunning;
    Ui::MainWindow *ui;
    TestForm* testForm;
    McaOGLWindow* openGlWidget;
};

#endif // MAINWINDOW_H
