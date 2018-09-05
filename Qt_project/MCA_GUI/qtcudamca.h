#ifndef QTCUDAMCA_H
#define QTCUDAMCA_H

#include "mainwindow.h"

class QtCudaMCA {
public:
    int run(int argc, char *argv[]);
protected:
     void getMainWindow(MainWindow* refW);
private:
    MainWindow* refW;
};

#endif // QTCUDAMCA_H
