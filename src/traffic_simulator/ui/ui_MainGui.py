# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainGuiIoMATh.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QMainWindow, QMenu,
    QMenuBar, QPlainTextEdit, QPushButton, QSizePolicy,
    QSpacerItem, QStatusBar, QTextBrowser, QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(999, 729)
        font = QFont()
        font.setStyleStrategy(QFont.PreferAntialias)
        MainWindow.setFont(font)
        MainWindow.setMouseTracking(False)
        MainWindow.setIconSize(QSize(20, 200))
        self.open_README = QAction(MainWindow)
        self.open_README.setObjectName(u"open_README")
        self.about_code = QAction(MainWindow)
        self.about_code.setObjectName(u"about_code")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.generate_report_button = QPushButton(self.centralwidget)
        self.generate_report_button.setObjectName(u"generate_report_button")

        self.gridLayout.addWidget(self.generate_report_button, 1, 2, 1, 1)

        self.graph_widget = PlotWidget(self.centralwidget)
        self.graph_widget.setObjectName(u"graph_widget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(35)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.graph_widget.sizePolicy().hasHeightForWidth())
        self.graph_widget.setSizePolicy(sizePolicy)
        self.graph_widget.setAutoFillBackground(False)

        self.gridLayout.addWidget(self.graph_widget, 0, 0, 1, 9)

        self.horizontalSpacer_2 = QSpacerItem(174, 17, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 1, 7, 1, 1)

        self.information_browser3 = QTextBrowser(self.centralwidget)
        self.information_browser3.setObjectName(u"information_browser3")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(3)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.information_browser3.sizePolicy().hasHeightForWidth())
        self.information_browser3.setSizePolicy(sizePolicy1)
        font1 = QFont()
        font1.setPointSize(10)
        font1.setStyleStrategy(QFont.PreferAntialias)
        self.information_browser3.setFont(font1)

        self.gridLayout.addWidget(self.information_browser3, 1, 8, 1, 1)

        self.accelerate_button = QPushButton(self.centralwidget)
        self.accelerate_button.setObjectName(u"accelerate_button")

        self.gridLayout.addWidget(self.accelerate_button, 1, 6, 1, 1)

        self.stop_or_start_button = QPushButton(self.centralwidget)
        self.stop_or_start_button.setObjectName(u"stop_or_start_button")

        self.gridLayout.addWidget(self.stop_or_start_button, 1, 3, 1, 2)

        self.decelerate_button = QPushButton(self.centralwidget)
        self.decelerate_button.setObjectName(u"decelerate_button")

        self.gridLayout.addWidget(self.decelerate_button, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(174, 23, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 1, 0, 1, 1)

        self.restart_button = QPushButton(self.centralwidget)
        self.restart_button.setObjectName(u"restart_button")

        self.gridLayout.addWidget(self.restart_button, 1, 5, 1, 1)

        self.information_browser2 = QTextBrowser(self.centralwidget)
        self.information_browser2.setObjectName(u"information_browser2")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(8)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.information_browser2.sizePolicy().hasHeightForWidth())
        self.information_browser2.setSizePolicy(sizePolicy2)
        font2 = QFont()
        font2.setPointSize(12)
        font2.setStyleStrategy(QFont.PreferAntialias)
        self.information_browser2.setFont(font2)

        self.gridLayout.addWidget(self.information_browser2, 2, 4, 1, 5)

        self.information_browser1 = QPlainTextEdit(self.centralwidget)
        self.information_browser1.setObjectName(u"information_browser1")
        self.information_browser1.setFont(font2)

        self.gridLayout.addWidget(self.information_browser1, 2, 0, 1, 4)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 999, 33))
        self.help_menu = QMenu(self.menubar)
        self.help_menu.setObjectName(u"help_menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.menubar.addAction(self.help_menu.menuAction())
        self.help_menu.addAction(self.open_README)
        self.help_menu.addAction(self.about_code)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"TrafficSimulator", None))
        self.open_README.setText(QCoreApplication.translate("MainWindow", u"\u4f7f\u7528\u8bf4\u660e", None))
#if QT_CONFIG(shortcut)
        self.open_README.setShortcut(QCoreApplication.translate("MainWindow", u"F3", None))
#endif // QT_CONFIG(shortcut)
        self.about_code.setText(QCoreApplication.translate("MainWindow", u"\u5173\u4e8e\u8fd9\u4e2a\u7a0b\u5e8f", None))
#if QT_CONFIG(shortcut)
        self.about_code.setShortcut(QCoreApplication.translate("MainWindow", u"F4", None))
#endif // QT_CONFIG(shortcut)
        self.generate_report_button.setText(QCoreApplication.translate("MainWindow", u"Gen Report", None))
        self.accelerate_button.setText(QCoreApplication.translate("MainWindow", u"Accelerate", None))
#if QT_CONFIG(shortcut)
        self.accelerate_button.setShortcut(QCoreApplication.translate("MainWindow", u"Up", None))
#endif // QT_CONFIG(shortcut)
        self.stop_or_start_button.setText(QCoreApplication.translate("MainWindow", u"Start/Stop", None))
#if QT_CONFIG(shortcut)
        self.stop_or_start_button.setShortcut(QCoreApplication.translate("MainWindow", u"Space", None))
#endif // QT_CONFIG(shortcut)
        self.decelerate_button.setText(QCoreApplication.translate("MainWindow", u"Decelerate", None))
#if QT_CONFIG(shortcut)
        self.decelerate_button.setShortcut(QCoreApplication.translate("MainWindow", u"Down", None))
#endif // QT_CONFIG(shortcut)
        self.restart_button.setText(QCoreApplication.translate("MainWindow", u"Restart", None))
        self.help_menu.setTitle(QCoreApplication.translate("MainWindow", u"\u5e2e\u52a9", None))
    # retranslateUi

