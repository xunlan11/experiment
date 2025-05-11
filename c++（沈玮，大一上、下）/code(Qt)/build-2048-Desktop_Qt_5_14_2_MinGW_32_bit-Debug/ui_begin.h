/********************************************************************************
** Form generated from reading UI file 'begin.ui'
**
** Created by: Qt User Interface Compiler version 5.14.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BEGIN_H
#define UI_BEGIN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Begin
{
public:
    QWidget *centralwidget;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *Begin)
    {
        if (Begin->objectName().isEmpty())
            Begin->setObjectName(QString::fromUtf8("Begin"));
        centralwidget = new QWidget(Begin);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        Begin->setCentralWidget(centralwidget);
        menubar = new QMenuBar(Begin);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 363, 25));
        Begin->setMenuBar(menubar);
        statusbar = new QStatusBar(Begin);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        Begin->setStatusBar(statusbar);

        retranslateUi(Begin);

        QMetaObject::connectSlotsByName(Begin);
    } // setupUi

    void retranslateUi(QMainWindow *Begin)
    {
        Begin->setWindowTitle(QCoreApplication::translate("Begin", "Begin", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Begin: public Ui_Begin {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BEGIN_H
