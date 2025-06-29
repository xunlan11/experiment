#include "begin.h"
#include "mainwindow.h"
#include "ui_begin.h"
#include "ui_mainwindow.h"

Begin::Begin(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::Begin)
{
    ui->setupUi(this);
    button1 = new QPushButton("进入游戏",this);                                 //跳转到游戏界面
    button1->setGeometry(190,400,80,50);
    connect(button1,SIGNAL(clicked()),this,SLOT(slot1()));
    button2 = new QPushButton("背景介绍",this);                                 //弹出背景介绍弹窗
    button2->setGeometry(50,400,80,50);
    connect(button2,SIGNAL(clicked()),this,SLOT(slot2()));
}

Begin::~Begin()
{
    delete ui;
}

void Begin::slot1()
{
    this->close();
    m = new MainWindow;
    m->setFixedSize(320,480);
    m->setWindowTitle("2048");
    m->show();
}

void Begin::slot2()
{
    QMessageBox::about(this,"背景介绍","      大沽口炮台位于天津大沽口海河南岸，是入京咽喉、津门屏障，乃“津门十景”中的“海门古塞”，素与虎门炮台并称。大沽口炮台几经兴废，融汇了西方的先进技术和设备，熔铸了中华儿女的智慧和坚毅，共建有5座主炮台，以“威”、“震”、“海”、“门”、“高”命名。在咸丰八年（1858年）、咸丰九年（1859年）、咸丰十年（1860年）、光绪二十六年（1900年），清军于此四次抵御西方侵略者，唯有第二次胜利。但是，大沽口炮台铭记了中华儿女抵抗侵略的血泪史，饱含罗荣光等将士以身殉国的悲壮，也见证了大沽铁钟的前世今生。");
}

void Begin::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.drawPixmap(rect(),QPixmap(":/background2.jpeg"),QRect());
    p.setBrush(Qt::blue);
    p.setFont(QFont("微软雅黑",10,700,false));
    QString strscore;
    p.drawText(QPoint(20,40),"大沽口炮台海防&2048");
    p.drawText(QPoint(20,100),"游戏规则：");
    p.drawText(QPoint(20,140),"①2到2048依次替换为：");
    p.drawText(QPoint(20,180),"抬枪、三合土、铁蒺藜、");
    p.drawText(QPoint(20,220),"水雷、克虏伯大炮、北塘炮台、");
    p.drawText(QPoint(20,260),"罗荣光、大沽铁钟、威字炮台、");
    p.drawText(QPoint(20,300),"威震海门高、海门古塞");
    p.drawText(QPoint(20,340),"②操作键位为wsad");
}
