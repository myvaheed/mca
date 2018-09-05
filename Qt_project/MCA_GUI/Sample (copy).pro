#-------------------------------------------------
#
# Project created by QtCreator 2017-07-11T14:58:06
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Sample
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    mcaoglwidget.cpp

HEADERS  += mainwindow.h \
    qt_mca_cuda.h \
    mcaoglwidget.h

FORMS    += mainwindow.ui
