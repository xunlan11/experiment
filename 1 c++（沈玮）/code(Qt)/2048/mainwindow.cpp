#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    for(int i=0;i<4;i++){                                                   //初始化各格子为0
        for(int j=0;j<4;j++){
            s[i][j]=0;
        }
    }
    button = new QPushButton("开始游戏",this);                               //开始游戏按钮
    button->setGeometry(60,400,200,50);
    connect(button,SIGNAL(clicked()),this,SLOT(slot()));
    qsrand(uint(QTime(0,0,0).secsTo(QTime::currentTime())));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *)
{
    QPainter p(this);                                                //对界面、计分器、格子等进行设计
    p.drawPixmap(rect(),QPixmap(":/background1.jpeg"),QRect());
    p.setBrush(Qt::blue);
    p.setFont(QFont("微软雅黑",20,700,false));
    QString strscore;
    p.drawText(QPoint(20,60),"分数: "+QString::number(score));

    for (int i=0;i<4;i++) {                                          //融入主题元素
        for (int j=0;j<4;j++) {
            p.setPen(Qt::transparent);
            if(s[i][j]==0){
                p.setBrush(QColor(255,250,222,255));
                p.drawRect(i*60+40,j*60+120,55,55);
            }
            else if (s[i][j]==2) {
                p.setBrush(QColor(255,229,168,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"抬枪",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==4) {
                p.setBrush(QColor(255,197,136,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"三合土",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==8) {
                p.setBrush(QColor(251,179,129,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"铁蒺藜",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==16) {
                p.setBrush(QColor(250,160,99,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"水雷",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==32) {
                p.setBrush(QColor(251,103,55,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"克虏伯大炮",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==64) {
                p.setBrush(QColor(213,92,14,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"北塘炮台",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==128) {
                p.setBrush(QColor(252,161,159,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"罗荣光",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==256) {
                p.setBrush(QColor(248,126,124,147));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"大沽铁钟",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==512) {
                p.setBrush(QColor(255,50,80,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"“威”字炮台",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==1024) {
                p.setBrush(QColor(188,25,22,255));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"威震海门高",QTextOption(Qt::AlignCenter));
            }
            else if (s[i][j]==2048) {
                p.setBrush(QColor(245,69,67,15));
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),"海门古塞",QTextOption(Qt::AlignCenter));
            }
            else{
                p.setBrush(Qt::darkBlue);
                p.drawRect(i*60+40,j*60+120,55,55);
                p.setPen(Qt::black);
                p.setFont(QFont("微软雅黑",10,700,false));
                p.drawText(QRectF(i*60+40,j*60+120,55,55),QString::number(s[i][j]),QTextOption(Qt::AlignCenter));
            }
        }
    }
}


void MainWindow::slot(){
    score=0;
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            s[i][j]=0;
        }
    }
    button->setText("重新游戏");

    int randi=qrand()%4;                       //随机开始位置
    int randj=qrand()%4;
    s[randi][randj]=2;

    state=true;
    update();
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    if(!state){                                //操作按键，只有wsad有效
        return ;
    }
        switch (event->key()) {
        case Qt::Key_W:
            Up();
            break;

        case Qt::Key_S:
            Down();
            break;

        case Qt::Key_A:
            Left();
            break;

        case Qt::Key_D:
            Right();
            break;

        default:
            return;
        }
        Rand();
        update();
}

void MainWindow::Up(){                         //四个方向的移动
    for (int i=0;i<4;i++) {
        for (int j=1;j<4;j++) {
            if(s[i][j]==0)continue;
            for (int p=0;p<j;p++) {            //查看是否有空格可移动
                if(s[i][p]==0){
                    s[i][p]=s[i][j];
                    s[i][j]=0;
                    break;
                }
            }
        }
    }
    for (int i=0; i<4; i++){                  //相加（2222->0444型和2244->0448型）
        for (int j=0; j<3; j++){
            if(s[i][j]==s[i][j+1]){
                s[i][j]=2*s[i][j];
                s[i][j+1]=0;
                score+=s[i][j];
                for (int p=j+2; p<4; p++){
                    if(p<3){
                        s[i][p-1]=s[i][p];
                    }
                    else{
                        s[i][p-1]=s[i][p];
                        s[i][p]=0;
                    }
                }
            }
        }
    }
}
void MainWindow::Down(){
    for (int i=0;i<4;i++) {
        for (int j=2;j>=0;j--) {
            if(s[i][j]==0)continue;
            for (int p=3;p>j;p--) {
                if(s[i][p]==0){
                    s[i][p]=s[i][j];
                    s[i][j]=0;
                    break;
                }
            }
        }
    }
    for(int i=0; i<4; i++){
        for(int j=3; j>0; j--){
            if(s[i][j]==s[i][j-1]){
                s[i][j]=2*s[i][j];
                s[i][j-1]=0;
                score+=s[i][j];
                for(int p=j-2; p>-1; p--){
                    if(p>0){
                        s[i][p+1]=s[i][p];
                    }
                    else{
                        s[i][p+1]=s[i][p];
                        s[i][p]=0;
                    }
                }
            }
        }
    }
}
void MainWindow::Left(){
    for (int j=0;j<4;j++) {
        for (int i=1;i<4;i++) {
            if(s[i][j]==0){
                continue;
            }
            for (int p=0;p<i;p++) {
                if(s[p][j] == 0){
                    s[p][j] = s[i][j];
                    s[i][j] = 0;
                    break;
                }
            }
        }
    }
    for(int j=0; j<4; j++){
        for(int i=0; i<3; i++){
            if(s[i][j]==s[i+1][j]){
                s[i][j]=2*s[i][j];
                s[i+1][j]=0;
                score+=s[i][j];
                for(int p=i+2; p<4; p++){
                    if(p<3){
                        s[p-1][j]=s[p][j];
                    }
                    else{
                        s[p-1][j]=s[p][j];
                        s[p][j]=0;
                    }
                }
            }
        }
    }
}
void MainWindow::Right(){
    for (int j=0;j<4;j++) {
        for (int i=2;i>=0;i--) {
            if(s[i][j]==0){
                continue;
            }
            for (int p=3;p>i;p--) {
                if(s[p][j] == 0){
                    s[p][j] = s[i][j];
                    s[i][j] = 0;
                    break;
                }
            }
        }
    }
    for(int j=0; j<4; j++){
        for(int i=3; i>0; i--){
            if(s[i][j]==s[i-1][j]){
                s[i][j]=2*s[i][j];
                s[i-1][j]=0;
                score+=s[i][j];
                for(int p=i-2; p>-1; p--){
                    if(p==1){
                        s[p+1][j]=s[p][j];
                    }
                    else{
                        s[p+1][j]=s[p][j];
                        s[p][j]=0;
                    }
                }
            }
        }
    }
}
void MainWindow::Rand(){
    int i=0,j=0;
    struct Ns n[15];
    int ni=0;                                     //判断为0的格子的数量
    for (i=0;i<4;i++) {
        for (j=0;j<4;j++) {
            if(s[i][j]==0){
                n[ni].i=i;
                n[ni].j=j;
                ni++;
            }
        }
    }
    if (ni==0) {                                  //格子全部为零时，判断能否进行合并，以判断游戏是否结束
        for (i=0;i<4;i++) {
            for (j=0;j<3;j++) {
                if(s[i][j]==s[i][j+1]){
                    return;
                }
            }
        }
        for (j=0;j<4;j++) {
            for (i=0;i<3;i++) {
                if(s[i][j]==s[i+1][j]){
                    return;
                }
            }
        }
        QMessageBox::about(this,"游戏失败","分数为："+QString::number(score)+" ");
        return;
    }
    int rand=qrand()%ni;                          //随机位置生成2
    s[n[rand].i][n[rand].j]=2;
}
