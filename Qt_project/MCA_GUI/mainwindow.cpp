#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

#include "math.h"
#include <QKeyEvent>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), isRunning(false)
{
    ui->setupUi(this);

    connect(ui->etNparts, SIGNAL(textChanged(QString)), SLOT(updateEnv()));
    connect(ui->etCountCol, SIGNAL(textChanged(QString)), SLOT(updateEnv()));
    connect(ui->etAutD, SIGNAL(textChanged(QString)), SLOT(updateEnv()));
    connect(ui->etAutD, SIGNAL(textChanged(QString)), SLOT(updateAutD()));
    connect(ui->etAutMass, SIGNAL(textChanged(QString)), SLOT(updateEnv()));
    connect(ui->cbPackMode, SIGNAL(currentIndexChanged(int)), SLOT(updateEnv()));

    connect(ui->etE, SIGNAL(textChanged(QString)), SLOT(updateConst()));
    connect(ui->etMu, SIGNAL(textChanged(QString)), SLOT(updateConst()));
    connect(ui->etViscosity, SIGNAL(textChanged(QString)), SLOT(updateConst()));
    connect(ui->etHmax, SIGNAL(textChanged(QString)), SLOT(updateConst()));

    connect(ui->cbForces, SIGNAL(currentIndexChanged(int)), SLOT(updateRun()));
    connect(ui->cbColorMode, SIGNAL(currentIndexChanged(int)), SLOT(updateRun()));
    connect(ui->slTimestep, SIGNAL(valueChanged(int)), SLOT(sliderTimestepChanged(int)));
    connect(ui->slDeltaForce, SIGNAL(valueChanged(int)), SLOT(sliderDeltaForceChanged(int)));
    connect(ui->slColliderSpring, SIGNAL(valueChanged(int)), SLOT(sliderColliderSpringChanged(int)));
    connect(ui->etTimestep, SIGNAL(textChanged(QString)), SLOT(updateRun()));
    connect(ui->etDeltaForce, SIGNAL(textChanged(QString)), SLOT(updateRun()));
    connect(ui->etColliderSpring, SIGNAL(textChanged(QString)), SLOT(updateRun()));
    connect(ui->chTransparency, SIGNAL(toggled(bool)), SLOT(updateRun()));

    connect(ui->cbNumThreads, SIGNAL(currentIndexChanged(int)), SLOT(updateTest()));
    connect(ui->cbHardware, SIGNAL(currentIndexChanged(int)), SLOT(hardwareTypeChanged(int)));
    connect(ui->etNumIterations, SIGNAL(textChanged(QString)), SLOT(updateTest()));

    connect(ui->btRun, SIGNAL(clicked(bool)), SLOT(startClick()));
    connect(ui->btStop, SIGNAL(clicked(bool)), SLOT(stopClick()));
    connect(ui->btnReset, SIGNAL(clicked(bool)), SLOT(resetClick()));

    connect(ui->btnTest, SIGNAL(clicked(bool)), SLOT(testClick()));

    initGUI();
}

void MainWindow::initGUI() {

    QStringList list;
    for (int i = 0; i < UIForceMode::UI_FORCEMODE_SIZE; i++) {
        UIForceMode mode = static_cast<UIForceMode>(i);
        list << UIForceModeStrings[mode];
    }
    ui->cbForces->addItems(list);

    list.clear();
    for (int i = 0; i < UIColorMode::UI_COLOR_MODE_SIZE; i++) {
        UIColorMode mode = static_cast<UIColorMode>(i);
        list << UIColorModeStrings[mode];
    }
    ui->cbColorMode->addItems(list);

    list.clear();
    for (int i = 0; i < UIPackMode::UI_PACK_MODE_SIZE; i++) {
        UIPackMode mode = static_cast<UIPackMode>(i);
        list << UIPackModeStrings[mode];
    }
    ui->cbPackMode->addItems(list);

    list.clear();
    for (int i = 0; i < UIHardwareType::UI_HARDWARE_TYPE_SIZE; i++) {
        UIHardwareType mode = static_cast<UIHardwareType>(i);
        list << UIHardwareTypeStrings[mode];
    }
    ui->cbHardware->addItems(list);

    ui->etTimestep->setValidator(new QDoubleValidator(0.0001, 1.0, 4, this));
    ui->etDeltaForce->setValidator(new QDoubleValidator(1.0, 1000.0, 4, this));
    ui->etColliderSpring->setValidator(new QDoubleValidator(1.0, 1000.0, 4, this));
    ui->etAutD->setValidator(new QDoubleValidator(0.0001, 1000, 4, this));
    ui->etAutMass->setValidator(new QDoubleValidator(0.0001, 10000, 4, this));

    ui->etHmax->setValidator(new QDoubleValidator(0.000001, 1000.0, 6, this));
    ui->etE->setValidator(new QDoubleValidator(0.000001, 100000.0, 6, this));
    ui->etMu->setValidator(new QDoubleValidator(0.000001, 100000.0, 6, this));
    ui->etViscosity->setValidator(new QDoubleValidator(0.000001, 100000.0, 6, this));

    ui->etNparts->setValidator(new QIntValidator(1, 10000000, this));
    ui->etCountCol->setValidator(new QIntValidator(1, 10000000, this));

    ui->etNumIterations->setValidator(new QIntValidator(1, 10000000, this));

    //init system
    ui->etNparts->setText(QString::number(3375));
    ui->etCountCol->setText(QString::number(15));
    ui->etCountOfRow->setText(QString::number(15));
    ui->etAutD->setText(QString::number(5));
    ui->etAutMass->setText(QString::number(1));

    ui->etE->setText(QString::number(70000));
    ui->etMu->setText(QString::number(0.34));
    ui->etViscosity->setText(QString::number(-1));

    sliderTimestepChanged(1);
    sliderDeltaForceChanged(0);
    sliderColliderSpringChanged(0);

    ui->cbPackMode->setCurrentIndex(0);
    ui->cbForces->setCurrentIndex(0);
    ui->cbColorMode->setCurrentIndex(0);

    ui->tvRealTime->setText(QString::number(0));
    ui->tvRunTime->setText(QString::number(0));
    ui->tvFPS->setText(QString::number(0));

    openGlWidget = 0;

    ui->cbHardware->setCurrentIndex(0);
    ui->etNumIterations->setText(QString::number(100));

    ui->hlViscosityParamsWidget->hide();
}

void MainWindow::updateEnv() {
    uint Nparts = ui->etNparts->text().toUInt();
    uint countOfCol = ui->etCountCol->text().toUInt();
    uint countOfRow = ui->etCountOfRow->text().toUInt();

    double automateD = ui->etAutD->text().toDouble();
    double automateM = ui->etAutMass->text().toDouble();
    UIPackMode packMode = static_cast<UIPackMode>(ui->cbPackMode->currentIndex());
    updatedEnv(Nparts, countOfCol, countOfRow, automateD, automateM, packMode);

    ui->tvAutV->setText(QString::number(getAutomateVolume()));
    ui->tvDensity->setText(QString::number(getDensity()));
    ui->tvMass->setText(QString::number(getTotalMass()));
    ui->tvV->setText(QString::number(getTotalVolume()));

    if (packMode == UIPackMode::PACK_GRID) {
        ui->etCountOfRow->setEnabled(false);
        ui->etCountOfRow->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        ui->tvCountOfRow->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    } else {
        ui->etCountOfRow->setEnabled(true);
        ui->etCountOfRow->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        ui->tvCountOfRow->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    }
}

void MainWindow::updateAutD() {
    double automateD = ui->etAutD->text().toDouble();
    ui->etHmax->setText(QString::number((automateD * 2.0/sqrt(3.0)) / 2.0 * (2.0 - sqrt(3.0))));
}

double MainWindow::getTotalMass() {
    return ui->etNparts->text().toUInt() * ui->etAutMass->text().toDouble();
}

double MainWindow::getTotalVolume() {
    return ui->etNparts->text().toUInt() * getAutomateVolume();
}

double MainWindow::getDensity() {
    if (!getTotalVolume())
        return 0;
    return getTotalMass() / getTotalVolume();
}

double MainWindow::getAutomateVolume() {
    double automateD = ui->etAutD->text().toDouble();
    return automateD * automateD * automateD * sqrt(3) / 2.0;
}


void MainWindow::updateConst() {
    double deltaH = ui->etHmax->text().toDouble();
    double YoungsModule = ui->etE->text().toDouble();
    double PoissionModule = ui->etMu->text().toDouble();
    double viscosity = ui->etViscosity->text().toDouble();
    updatedConst(deltaH, YoungsModule, PoissionModule, viscosity);
    ui->tvG->setText(QString::number(getShearModulus()));
    ui->tvK->setText(QString::number(getBulkModulus()));
}

double MainWindow::getShearModulus() {
    double YoungsModule = ui->etE->text().toDouble();
    double PoissionModule = ui->etMu->text().toDouble();
    return YoungsModule / (2.0 * (1.0 + PoissionModule));
}

double MainWindow::getBulkModulus() {
    double YoungsModule = ui->etE->text().toDouble();
    double PoissionModule = ui->etMu->text().toDouble();
    if (PoissionModule == 0.5)
        return 0;
    return YoungsModule / (3.0 * (1.0 - 2 * PoissionModule));
}

void MainWindow::updateRun() {
    double timestep = ui->etTimestep->text().toDouble();
    double deltaForce = ui->etDeltaForce->text().toDouble();
    double colliderSpring = ui->etColliderSpring->text().toDouble();
    bool transperency = ui->chTransparency->isChecked();

    UIForceMode forceMode = static_cast<UIForceMode>(ui->cbForces->currentIndex());
    UIColorMode colorMode = static_cast<UIColorMode>(ui->cbColorMode->currentIndex());

    updatedRunMode(timestep, deltaForce, colliderSpring, transperency, forceMode, colorMode);

    ui->slTimestep->setValue(timestep * ui->slTimestep->maximum());
    ui->slDeltaForce->setValue(deltaForce);
    ui->slColliderSpring->setValue(colliderSpring);
}

int MainWindow::getNumThreads(UIHardwareType hardwareType) {
    UINumThreadsCPU numThreadsCPU;
    UINumThreadsGPU numThreadsGPU;

    //numThreads == -1 - ALL
    int numThreads = 0;
    if (hardwareType == HARDWARE_CPU) {
        numThreadsCPU = static_cast<UINumThreadsCPU>(ui->cbNumThreads->currentIndex());
        switch (numThreadsCPU) {
        case THREADS_CPU_ALL:
            numThreads = 0;
            break;
        case THREADS_CPU_1:
            numThreads = 1;
            break;
        case THREADS_CPU_2:
            numThreads = 2;
            break;
        case THREADS_CPU_4:
            numThreads = 4;
            break;
        default:
            numThreads = 0;
            break;
        }
    } else {
        numThreadsGPU = static_cast<UINumThreadsGPU>(ui->cbNumThreads->currentIndex());
        switch (numThreadsGPU) {
        case THREADS_GPU_ALL:
            numThreads = 0;
            break;
        case THREADS_GPU_16:
            numThreads = 16;
            break;
        case THREADS_GPU_32:
            numThreads = 32;
            break;
        case THREADS_GPU_64:
            numThreads = 64;
            break;
        case THREADS_GPU_128:
            numThreads = 128;
            break;
        case THREADS_GPU_256:
            numThreads = 256;
            break;
        case THREADS_GPU_512:
            numThreads = 512;
            break;
        case THREADS_GPU_1024:
            numThreads = 1024;
            break;
        case THREADS_GPU_2048:
            numThreads = 2048;
            break;
        case THREADS_GPU_5096:
            numThreads = 5096;
            break;
        default:
            numThreads = 0;
            break;
        }
    }
    return numThreads;
}

void MainWindow::hardwareTypeChanged(int value) {
    UIHardwareType hardwareType = static_cast<UIHardwareType>(value);
    QStringList list;
    ui->cbNumThreads->clear();
    if (hardwareType == HARDWARE_CPU) {
        for (int i = 0; i < UINumThreadsCPU::UI_NUM_THREADS_CPU_SIZE; i++) {
            UINumThreadsCPU mode = static_cast<UINumThreadsCPU>(i);
            list << UINumThreadsCPUStrings[mode];
        }
    } else {
        for (int i = 0; i < UINumThreadsGPU::UI_NUM_THREADS_GPU_SIZE; i++) {
            UINumThreadsGPU mode = static_cast<UINumThreadsGPU>(i);
            list << UINumThreadsGPUStrings[mode];
        }
    }
    ui->cbNumThreads->addItems(list);
    ui->cbNumThreads->setCurrentIndex(0);
}

void MainWindow::updateTest() {
    std::cout << " updateTest " << std::endl;
    int numIterations = ui->etNumIterations->text().toInt();
    UIHardwareType hardwareType = static_cast<UIHardwareType>(ui->cbHardware->currentIndex());

    int numThreads = getNumThreads(hardwareType);
    updatedTest(hardwareType, numThreads, numIterations);
}



void MainWindow::glGUIDrawStep(double realtimeSeconds, double runtimeSeconds, double fps) {
    ui->tvRealTime->setText(QString::number(realtimeSeconds));
    ui->tvRunTime->setText(QString::number(runtimeSeconds));
    ui->tvFPS->setText(QString::number(fps));
}

void MainWindow::startClick() {
    if (startClicked()) {
        onStart();
    } else {
        onPause();
    }
}

void MainWindow::stopClick() {
    if (stopClicked()) {
        onStop();
        return;
    }
}

void MainWindow::resetClick() {
    if (resetClicked()) {
        stopClick();
        startClick();
        return;
    }
}

void MainWindow::testClick() {
    if (testClicked()) {
        ui->btnTest->setText("Stop Test");
        onStart();
    } else {
        if (stopClicked()) {
            ui->btnTest->setText("Test");
            onStop();
        }
    }
}

void MainWindow::sliderTimestepChanged(int value) {
    double timestep = (double) value / ui->slTimestep->maximum();
    ui->etTimestep->setText(QString::number(timestep));
}

void MainWindow::sliderDeltaForceChanged(int value) {
    double deltaForce = (double) value;
    ui->etDeltaForce->setText(QString::number(deltaForce));
}

void MainWindow::sliderColliderSpringChanged(int value) {
    double colliderSpring = (double) value;
    ui->etColliderSpring->setText(QString::number(colliderSpring));
}

//void MainWindow::updatedEnv(int Nparts, int countOfCol, int countOfRow, double automateD, double automateM, UIPackMode mode) {
//    std::cout << Nparts << std::endl;
//}

//void MainWindow::updatedConst(double deltaH, double mcaE, double mcaMu, double mcaViscosity) {

//}

//void MainWindow::updatedRunMode(double timestep, double deltaForce, double collierSpring, bool transparency, UIForceMode forceMode, UIColorMode colorMode) {
//    std::cout << " updatedRunMode " << " forceMode = " << UIForceModeStrings[forceMode] << std::endl;
//}

//void MainWindow::updatedTest(UIHardwareType hardwareType, int numThreads, int numIterations) {
//    std::cout << "hardware type = " << UIHardwareTypeStrings[hardwareType] << " numThreads = " << numThreads << std::endl;
//    std::cout << " num Iterations = " << numIterations << std::endl;
//}

//bool MainWindow::startClicked() {
//    std::cout << "run" << std::endl;
//    return !isRunning;
//}

//bool MainWindow::stopClicked() {
//    std::cout << "stop" << std::endl;
//    return isRunning;
//}

//bool MainWindow::resetClicked() {

//}

//bool MainWindow::testClicked() {
//    std::cout << "test" << std::endl;
//    return !isRunning;
//}

void MainWindow::onStart() {
    isRunning = true;
    ui->btRun->setText("Pause");
    //ui->btnFullscreen->setVisible(true);

    ui->tabWidget->setTabEnabled(0, false);
    ui->tabWidget->setTabEnabled(1, false);

    //from pause
    if (openGlWidget) {
        return;
    }

    openGlWidget = new McaOGLWindow(this);
    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setStencilBufferSize(8);
    openGlWidget->setFormat(fmt);
    openGlWidget->showFullScreen();
}

void MainWindow::onPause() {
    isRunning = false;
    ui->btRun->setText("Run");
}

void MainWindow::onStop() {
    isRunning = false;
    ui->btRun->setText("Run");
    ui->tabWidget->setTabEnabled(0, true);
    ui->tabWidget->setTabEnabled(1, true);

    if (openGlWidget) {
        openGlWidget->close();
        delete openGlWidget;
        openGlWidget = 0;
    }
}

MainWindow::~MainWindow()
{
    onStop();
    delete ui;
}

