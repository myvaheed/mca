#include "mcaoglwidget.h"
#include <iostream>

#include <qopengl.h>
#include <QMouseEvent>


McaOGLWindow::McaOGLWindow(QWidget* parent):QOpenGLWindow(QOpenGLWindow::NoPartialUpdate)
{

}

McaOGLWindow::~McaOGLWindow()
{

}

void McaOGLWindow::keyPressEvent(QKeyEvent *event) {
   switch ( event->key() ) {
        case Qt::Key_V:
            //act on 'X'
            V_Pressed();
            break;
        case Qt::Key_M:
            //act on 'Y'
            M_Pressed();
            break;
         case Qt::Key_Escape:
       //act on 'Y'
            std::cout << "escape" << std::endl;
            //showNormal();
            break;
        case Qt::Key_Space:
       //act on 'Y'
            spacePressed();
            break;
        default:
            event->ignore();
            break;
        }
}


void McaOGLWindow::mouseMoveEvent(QMouseEvent *event) {
    switch (event->buttons()) {
        case Qt::LeftButton:
            leftButtonMotion(event->pos().x(), event->pos().y());
        break;
        case Qt::MiddleButton:
            middleButtonMotion(event->pos().x(), event->pos().y());
        break;
        case Qt::RightButton:
            rightButtonMotion(event->pos().x(), event->pos().y());
        break;
    }
}

void McaOGLWindow::mousePressEvent(QMouseEvent *event) {
    mousePressed(event->pos().x(), event->pos().y());
    switch (event->buttons()) {
        case Qt::RightButton:
            rightButtonClicked(event->pos().x(), event->pos().y());
        break;
    }
}

void McaOGLWindow::wheelEvent(QWheelEvent *event) {
    if (event->delta() > 0)
        zoomButtonMotion(true);
    else
        zoomButtonMotion(false);
}

//void McaOGLWindow::leftButtonMotion(int x, int y) {
//    std::cout << "left" << x << y <<  std::endl;
//}

//void McaOGLWindow::rightButtonMotion(int x, int y) {
//    std::cout << "right" << x << y << std::endl;
//}

//void McaOGLWindow::rightButtonClicked(int x, int y) {
//    std::cout << "right click" << x << y << std::endl;
//}

//void McaOGLWindow::middleButtonMotion(int x, int y) {
//    std::cout << "middle" << x << y <<  std::endl;
//}

//void McaOGLWindow::zoomButtonMotion(bool zoom) {
//    std::cout << "zoom" << zoom <<  std::endl;
//}

//void McaOGLWindow::mousePressed(int x, int y) {

//}

//void McaOGLWindow::spacePressed() {

//}
//void McaOGLWindow::V_Pressed() {

//}
//void McaOGLWindow::M_Pressed() {

//}



//void McaOGLWindow::initializeGL()
//{
//    std::cout << "initGL " << std::endl;
//    initializeOpenGLFunctions();
//    glClearColor(0,0,0,1);
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_LIGHT0);
//    glEnable(GL_LIGHTING);
//    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
//    glEnable(GL_COLOR_MATERIAL);
//}

//void McaOGLWindow::paintGL()
//{
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    glBegin(GL_TRIANGLES);
//        glColor3f(1.0, 0.0, 0.0);
//        glVertex3f(-0.5, -0.5, 0);
//        glColor3f(0.0, 1.0, 0.0);
//        glVertex3f( 0.5, -0.5, 0);
//        glColor3f(0.0, 0.0, 1.0);
//        glVertex3f( 0.0,  0.5, 0);
//    glEnd();
//}

//void McaOGLWindow::resizeGL(int w, int h)
//{
//    glViewport(0,0,w,h);
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    //gluPerspective(45, (float)w/h, 0.01, 100.0);
//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
//    //gluLookAt(0,0,5,0,0,0,0,1,0);
//}
