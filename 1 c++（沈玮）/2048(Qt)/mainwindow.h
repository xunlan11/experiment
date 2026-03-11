#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include <QKeyEvent>
#include <QPushButton>
#include <QPainter>
#include <QTime>
#include <QMessageBox>

namespace Ui {
class MainWindow;
}

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void paintEvent(QPaintEvent *);
    void keyPressEvent(QKeyEvent *event);
    void Up();
    void Down();
    void Left();
    void Right();
    void Rand();

    QPushButton *button;
    int s[4][4];
    int score=0;
    bool state;

    struct Ns{
        int i;
        int j;
    };

public slots:
        void slot();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
