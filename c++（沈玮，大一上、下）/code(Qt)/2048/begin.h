#ifndef BEGIN_H
#define BEGIN_H

#include "mainwindow.h"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class Begin;
}
QT_END_NAMESPACE

class Begin : public QMainWindow
{
    Q_OBJECT

public:
    Begin(QWidget *parent = nullptr);
    ~Begin();
    void paintEvent(QPaintEvent *);

    QPushButton *button1;
    QPushButton *button2;

private slots:
    void slot1();
    void slot2();

private:
    Ui::Begin *ui;
    MainWindow *m = new MainWindow;
};
#endif // BEGIN_H
