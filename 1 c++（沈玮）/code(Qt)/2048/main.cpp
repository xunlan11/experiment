#include "begin.h"
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Begin b;
    b.setFixedSize(320,480);
    b.setWindowTitle("2048");
    b.show();
    return a.exec();
}
