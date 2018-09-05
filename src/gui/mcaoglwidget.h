#ifndef MCAOGLWIDGET_H
#define MCAOGLWIDGET_H

#include <QtWidgets/qwidget.h>
#include <QtGui/qopenglwindow.h>
#include <QtGui/qopenglfunctions.h>

class McaOGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
{
public:
    McaOGLWindow(QWidget *parent = 0);
    ~McaOGLWindow();

protected:
    void leftButtonMotion(int x, int y);
    void middleButtonMotion(int x, int y);
    void rightButtonMotion(int x, int y);
    void rightButtonClicked(int x, int y);
    void zoomButtonMotion(bool zoom);
    void spacePressed();
    void V_Pressed();
    void M_Pressed();
    void mousePressed(int x, int y);

    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

private:
    virtual void keyPressEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
};




#endif // MCAOGLWIDGET_H
