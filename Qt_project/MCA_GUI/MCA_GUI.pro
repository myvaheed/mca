#-------------------------------------------------
#
# Project created by QtCreator 2017-07-11T14:58:06
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MCA_GUI
#TEMPLATE = app
TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++11


SOURCES += main.cpp\
        mainwindow.cpp \
    mcaoglwidget.cpp \
    qtcudamca.cpp

HEADERS  += mainwindow.h \
    qt_mca_cuda.h \
    mcaoglwidget.h \
    qtcudamca.h

FORMS    += mainwindow.ui

#unix {
#    target.path = /usr/lib
 #   INSTALLS += target
#}
