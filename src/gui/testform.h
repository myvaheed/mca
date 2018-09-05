#ifndef TESTFORM_H
#define TESTFORM_H

#include "Qt/qmainwindow.h"

namespace Ui {
class TestForm;
}

class TestForm : public QMainWindow
{
    Q_OBJECT
protected:
    void closeEvent(QCloseEvent *event);
public:
    explicit TestForm(QWidget *parent = 0);
    ~TestForm();
    void testFinished(double processedTime);
public slots:
    void stopTestClicked();
private:
    Ui::TestForm *ui;
};

#endif // TESTFORM_H
