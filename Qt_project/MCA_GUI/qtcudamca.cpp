#include "qtcudamca.h"
#include <QApplication>

//void QtCudaMCA::getMainWindow(MainWindow *refW) {

//}

int QtCudaMCA::run(int argc, char *argv[]) {
    QApplication a(argc, argv);
    MainWindow w;
    refW = &w;
    getMainWindow(refW);
    w.show();
    return a.exec();
}

